import torch
from torch import nn
from evoformer.dropout import DropoutRowwise, DropoutColumnwise
from evoformer.mha import MultiHeadAttention

class TriangleMultiplication(nn.Module):

    def __init__(self, c_z, mult_type, c=128):
        super().__init__()
        if mult_type not in {'outgoing', 'incoming'}:
            raise ValueError(f'mult_type must be either "outgoing" or "incoming" but is {mult_type}.')

        self.mult_type = mult_type
        self.layer_norm_in = nn.LayerNorm(c_z)
        self.layer_norm_out = nn.LayerNorm(c)
        self.linear_a_p = nn.Linear(c_z, c)
        self.linear_a_g = nn.Linear(c_z, c)
        self.linear_b_p = nn.Linear(c_z, c)
        self.linear_b_g = nn.Linear(c_z, c)
        self.linear_g = nn.Linear(c_z, c_z)
        self.linear_z = nn.Linear(c, c_z)

    def forward(self, z):
        out = None
        z = self.layer_norm_in(z)
        a = torch.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = torch.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)
        g = torch.sigmoid(self.linear_g(z))

        if self.mult_type == 'outgoing':
            z = torch.einsum('...ikc,...jkc->...ijc', a, b)
        else:
            z = torch.einsum('...kic,...kjc->...ijc', a, b)

        out = g * self.linear_z(self.layer_norm_out(z))
        
        return out

class TriangleAttention(nn.Module):
    def __init__(self, c_z, node_type, c=32, N_head=4):
        super().__init__()
        if node_type not in {'starting_node', 'ending_node'}:
            raise ValueError(f'node_type must be either "starting_node" or "ending_node" but is {node_type}')

        self.node_type = node_type

        self.layer_norm = nn.LayerNorm(c_z)
        if node_type == 'starting_node':
            attn_dim = -2
        else:
            attn_dim = -3

        self.mha = MultiHeadAttention(c_z, c, N_head, attn_dim, gated=True)
        self.linear = nn.Linear(c_z, N_head, bias=False)

    def forward(self, z):
        out = None

        z = self.layer_norm(z)
        bias = self.linear(z)
        bias = bias.moveaxis(-1, -3)
        if self.node_type == 'ending_node':
            bias = bias.transpose(-1, -2)

        out = self.mha(z, bias=bias)
        return out

class PairTransition(nn.Module):
    def __init__(self, c_z, n=4):
        super().__init__()

        self.layer_norm = nn.LayerNorm(c_z)
        self.linear_1 = nn.Linear(c_z, n*c_z)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n*c_z, c_z)

    def forward(self, z):
        out = None
        z = self.layer_norm(z)
        a = self.linear_1(z)
        out = self.linear_2(self.relu(a))

        return out


class PairStack(nn.Module):

    def __init__(self, c_z):
        super().__init__()

        self.dropout_rowwise = DropoutRowwise(p=0.25)
        self.dropout_columnwise = DropoutColumnwise(p=0.25)
        self.tri_mul_out = TriangleMultiplication(c_z, mult_type='outgoing')
        self.tri_mul_in = TriangleMultiplication(c_z, mult_type='incoming')
        self.tri_att_start = TriangleAttention(c_z, node_type='starting_node')
        self.tri_att_end = TriangleAttention(c_z, node_type='ending_node')
        self.pair_transition = PairTransition(c_z)

    def forward(self, z):

        out = None

        z = z + self.dropout_rowwise(self.tri_mul_out(z))
        z = z + self.dropout_rowwise(self.tri_mul_in(z))
        z = z + self.dropout_rowwise(self.tri_att_start(z))
        z = z + self.dropout_columnwise(self.tri_att_end(z))
        z = z + self.pair_transition(z)

        out = z

        return out
        