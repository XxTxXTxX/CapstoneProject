import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    
    def __init__(self, dim):
        '''
        Args:
            dim: Dimension of each head, usually represents c in MultiHeadAttention
        '''
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        '''
        Args:
            seq_len: N_res --> Length of sequence
            device
        '''
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i, j->ij', t, self.inv_freq)
        embedding = torch.cat((freqs, freqs), dim = -1)
        return embedding
    
def rotate_half(x):
    '''
    Partial rotary
    '''
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_embedding(q, k, cos, sin):
    '''
    Applaying rotary embedding with query and key

    Args:
        q: query tensor [batch, n_head, seq_len, head_dim]
        k: key tensor [batch, n_head, seq_len, head_dim]
        cos: cosine of otary embedding
        sin: sine of rotary embedding
    '''
    # make sure cos / sin has same dimension with query and key
    cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    