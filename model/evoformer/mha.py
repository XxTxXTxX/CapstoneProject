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

    def __init__(self, c_in, c, N_head, attn_dim, gated=False, is_global=False, use_bias_for_embeddings=False):
        """
        c_in: dmodel
        c: dk
        N_head: h
        """
        super().__init__()
        
        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        # if gated = True, initialize linear with bias
        self.gated = gated
        # which axis to apply attention across
        self.attn_dim = attn_dim
        self.is_global = is_global
        self.rotary = RotaryEmbedding(c)

        # Output layer uses bias
        # dmodel -> h*d_k
        self.linear_q = nn.Linear(c_in, c*N_head, bias=use_bias_for_embeddings)
        
        """
        if is_global = False, standard mha
        if is_global = True, attention using single shared KV
        """
        c_kv = c if is_global else c*N_head
        # False: dmodel -> h*d_k, True: dmodel -> dmodel
        self.linear_k = nn.Linear(c_in, c_kv, bias=use_bias_for_embeddings)
        self.linear_v = nn.Linear(c_in, c_kv, bias=use_bias_for_embeddings)
        
        # h*d_k -> dmodel
        self.linear_o = nn.Linear(c*N_head, c_in)

        if gated:
            # dmodel -> h*d_k
            self.linear_g = nn.Linear(c_in, c*N_head)

    def prepare_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # attn_dim usually = 1

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
        # q: batch, seq_len, h*dk
        # k,v: batch, seq_len, dmodel
        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)
        
        # q: batch, seq_len, h, d_k
        q_shape = q.shape[:-1] + (self.N_head, self.c)
        q = q.view(q_shape)
        # q: batch, h, seq_len, d_k
        q = q.transpose(-2, -3)
        # kv: batch, 1, seq_len, dmodel
        k = k.unsqueeze(-3)
        v = v.unsqueeze(-3)
        # q: batch, h, 1, d_k
        q = torch.mean(q, dim=-2, keepdim=True)

        return q, k, v

    def forward(self, x, bias=None, attention_mask=None):

        out = None

        # batch, seq_len, dmodel -> batch, seq_len, h*d_k
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if self.is_global:
            # q: batch, h, 1, d_k
            # kv: batch, 1, seq_len, dmodel (shared kv across query heads)
            q, k, v = self.prepare_qkv_global(q, k, v)
        else:
            # batch, h, seq_len, d_k
            q, k, v = self.prepare_qkv(q, k, v)

        # Apply rotary embedding
        seq_len = q.shape[-2]  # get sequence length
        rotary_pos_emb = self.rotary(seq_len, x.device)
        cos, sin = rotary_pos_emb.cos(), rotary_pos_emb.sin()
        q, k = apply_rotary_pos_embedding(q, k, cos, sin)

        # if is_global = False, q: batch, seq_len, h, d_k
        # if is_global = True, q: batch, h, 1, d_k
        q = q / math.sqrt(self.c)
        
        # 
        a = torch.einsum('...qc,...kc->...qk', q, k)
        if bias is not None:
            bias_batch_shape = bias.shape[:-3]
            bias_bc_shape = bias_batch_shape + \
                (1,) * (a.ndim-len(bias_batch_shape)-3) + bias.shape[-3:]
            bias = bias.view(bias_bc_shape)

            a = a + bias

        if attention_mask is not None:
            attention_mask = attention_mask[..., None, None, :]
            offset = (attention_mask == 0) * -1e8
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
