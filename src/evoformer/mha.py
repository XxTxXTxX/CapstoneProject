import torch
import math
from torch import nn
from evoformer.rotaryEmbedding import RotaryEmbedding, apply_rotary_pos_embedding

rotated = 0

def isRotated():
    return rotated

def plusOne():
    global rotated
    rotated += 1


class MultiHeadAttention(nn.Module):
    """
    A MultiHeadAttention module with optional bias and optional gating.
    """

    def __init__(self, c_in, c, N_head, attn_dim, gated=False, is_global=False, use_bias_for_embeddings=False):
        """
        Args:
            - c_in (int): Input dimension for the embeddings.
            - c (int): Embedding dimension for each individual head.
            - N_head (int): Number of heads.
            - attn_dim (int): The dimension in the input tensor along which the attention mechanism is performed.
            - gated (bool, optional): If True, an additional sigmoid-activated linear layer will be multiplicated against the weighted value vectors before feeding them through the output layer. Defaults to False.
            - is_global (bool, optional): If True, global calculation will be performed.
                For global calculation, key and value embeddings will only use one head,
                and the q query vectors will be averaged to one query vector.
                Defaults to False.
            use_bias_for_embeddings (bool, optional): If True, query, 
                key, and value embeddings will use bias, otherwise not. 
                Defaults to False.
        """
        super().__init__()

        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        self.gated = gated
        self.attn_dim = attn_dim
        self.is_global = is_global
        self.rotary = RotaryEmbedding(c)

        # Whether or not query, key, and value layers use bias is determined by `use_bias` (False for AlphaFold). 
        
        # The output layer should always use a bias. If gated is true, initialize another linear with bias.

        self.linear_q = nn.Linear(c_in, c*N_head, bias=use_bias_for_embeddings)

        c_kv = c if is_global else c*N_head
        self.linear_k = nn.Linear(c_in, c_kv, bias=use_bias_for_embeddings)
        self.linear_v = nn.Linear(c_in, c_kv, bias=use_bias_for_embeddings)

        self.linear_o = nn.Linear(c*N_head, c_in)

        if gated:
            self.linear_g = nn.Linear(c_in, c*N_head)

    def prepare_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # batch, seq_len, h*d_k -> batch, seq_len, h*d_k
        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)

        # batch, seq_len, h, -1 = d_k
        q_shape = q.shape[:-1] + (self.N_head, -1)
        k_shape = k.shape[:-1] + (self.N_head, -1)
        v_shape = v.shape[:-1] + (self.N_head, -1)

        # batch, seq_len, h, d_k
        q = q.view(q_shape)
        k = k.view(k_shape)
        v = v.view(v_shape)

        # batch, h, seq_len, d_k
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        return q, k, v

    def prepare_qkv_global(self, q, k, v):
        """
        Prepares the query, key and value embeddings with the following 
        differences to the non-global version:
            - key and value embeddings use only one head.
            - the query vectors are contracted into one, average query vector.
        
        Args:
            q (torch.tensor): Query embeddings of shape (*, q, *, N_head*c).
            k (torch.tensor): Key embeddings of shape (*, k, *, c).
            v (torch.tensor): Value embeddings of shape (*, v, *, c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of
                shape (*, N_head, 1, c) for q and shape (*, 1, k, c) for k and v. 
        """
        # batch, seq_len, h*dk
        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)

        q_shape = q.shape[:-1] + (self.N_head, self.c)
        q = q.view(q_shape)

        q = q.transpose(-2, -3)
        k = k.unsqueeze(-3)
        v = v.unsqueeze(-3)

        q = torch.mean(q, dim=-2, keepdim=True)

        return q, k, v

    def forward(self, x, bias=None, attention_mask=None):
        """
        Forward pass through the MultiHeadAttention module.

        Args:
            x: batch, seq_len, dmodel
            bias (torch.tensor, optional): Optional bias tensor of shape (batch, h, seq_len, seq_len) that will be added to the attention weights. Defaults to None.
            attention_mask (torch.tensor, optional): Optional attention mask of shape (*, k). If set, the keys with value 0 in the mask will not be attended to.

        Returns:
            torch.tensor: Output tensor of shape (*, q/k/v, *, c_in)
        """

        out = None

        """
        Implement the forward pass consisting of the following steps:
            - Create query, key and value embeddings.
            - Rearrange the embeddings with prepare_qkv
            - Scale the queries by 1/sqrt(c).
            - Calculate the attention weights of shape (batch, h, seq_len, seq_len) from q and k.
            - If a bias was given:
               - extract the bias batch shape by omitting the last 3 dims from bias.
        #       - construct a broadcastable bias shape, by concatenating         #
        #           bias_batch_shape, (1,) * n, and the last three dims of bias. #
        #           Choose n such that the broadcastable shape has as many dims  #
        #           as the attention scores.                                     #
        #       - add the bias to the attention scores.                          #
        #   - If an attention mask was given (not needed for AlphaFold):         #
        #       - unsqueeze the mask to make it broadcastable against the        #
        #         attention scores of shape (*, N_head, q, k).                   #
        #       - create a tensor `offset`` of the same shape as the mask with   #
        #         the value -1e8 where the mask is 0 and zero elsewhere.         #
        #       - add the offset to the raw attention scores.                    #
        #   - Use softmax to convert the attention scores into a                 #
        #       probability distribution.                                        #
        #   - Weight the value vectors by the attention weights and sum          #
        #       them up along the key dimension. You can use torch.einsum        #
        #       to do this in one line. The result should be                     #
        #       of shape (*, N_head, q, c).                                      #
        #   - Rearrange the intermediate output in the following way:            #
        #       * (*, N_head, q, c) -> (*, q, N_head, c)                         #
        #       * (*, q, N_head, c) -> (*, q, N_head * c)                        #
        #       * (*, q, N_head * c) -> (*, q, *, N_head * c)                    #
        #       The order of these transformations is crucial, as moving q       #
        #       to attn_dim before flattening the heads will result in an        #
        #       incorrect positioning if attn_dim uses negative indexing.        #
        #   - if gated, calculate the gating with linear_g and sigmoid and       #
        #       multiply it against the output.                                  #
        #   - apply linear_o to calculate the final output.
        """
        # batch, seq_len, dmodel -> batch, seq_len, h*d_k
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if self.is_global:
            # q: batch, h, seq_len = 1, d_k
            # 
            q, k, v = self.prepare_qkv_global(q, k, v)
        else:
            # batch, h, seq_len, d_k
            q, k, v = self.prepare_qkv(q, k, v)

        # Apply rotary embedding
        if isRotated <= 2:
            seq_len = q.shape[-2]  # get sequence length
            rotary_pos_emb = self.rotary(seq_len, x.device)
            cos, sin = rotary_pos_emb.cos(), rotary_pos_emb.sin()
            q, k = apply_rotary_pos_embedding(q, k, cos, sin)
            print(f"Rotated: {rotated}")
            plusOne()

        # 
        q = q / math.sqrt(self.c)

        a = torch.einsum('...qc,...kc->...qk', q, k)
        if bias is not None:
            bias_batch_shape = bias.shape[:-3]
            bias_bc_shape = bias_batch_shape + (1,) * (a.ndim-len(bias_batch_shape)-3) + bias.shape[-3:]
            bias = bias.view(bias_bc_shape)

            a = a + bias

        if attention_mask is not None:
            attention_mask = attention_mask[..., None, None, :]
            offset = (attention_mask==0) * -1e8
            a = a + offset

        a = torch.softmax(a, dim=-1)
        # o has shape [*, N_head, q, c]
        o = torch.einsum('...qk,...kc->...qc', a, v)
        o = o.transpose(-3, -2)
        o = torch.flatten(o, start_dim=-2)
        o = o.moveaxis(-2, self.attn_dim)
        if self.gated:
            g = torch.sigmoid(self.linear_g(x))
            o = g * o

        out = self.linear_o(o)

        return out