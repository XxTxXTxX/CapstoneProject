import torch
from torch import nn
from evoformer.mha import MultiHeadAttention

class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, c_m, c_z, c=32, N_head=1):
        super().__init__()
        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, N_head, bias=False)
        self.mha = MultiHeadAttention(c_m, c, N_head, attn_dim=-2, gated=True)

    def forward(self, m, z):

        out = None
        m = self.layer_norm_m(m)
        b = self.linear_z(self.layer_norm_z(z))
        b = b.moveaxis(-1, -3)
        out = self.mha(m, bias=b)

        return out

class MSAColumnAttention(nn.Module):

    def __init__(self, c_m, c=32, N_head=1):
        super().__init__()

        self.layer_norm_m = nn.LayerNorm(c_m)
        self.mha = MultiHeadAttention(c_m, c, N_head, attn_dim=-3, gated=True)

    def forward(self, m):

        out = None

        m = self.layer_norm_m(m)
        out = self.mha(m)

        return out


class MSATransition(nn.Module):
    def __init__(self, c_m, n=4):
        super().__init__()

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, n*c_m)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n*c_m, c_m)

    def forward(self, m):
        out = None

        m = self.layer_norm(m)
        a = self.linear_1(m)
        out = self.linear_2(self.relu(a))

        return out

class OuterProductMean(nn.Module):
    def __init__(self, c_m, c_z, c=32):
        super().__init__()

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c)
        self.linear_2 = nn.Linear(c_m, c)
        self.linear_out  = nn.Linear(c*c, c_z)

    def forward(self, m):
        N_seq = m.shape[-3]
        z = None

        m = self.layer_norm(m)
        a = self.linear_1(m)
        b = self.linear_2(m)
        o = torch.einsum('...sic,...sjd->...ijcd', a, b)
        o = torch.flatten(o, start_dim=-2)
        z = self.linear_out(o) / N_seq

        return z