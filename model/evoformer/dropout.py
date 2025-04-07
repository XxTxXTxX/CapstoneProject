import torch
from torch import nn

"""
drop entire rows or columns
"""

class SharedDropout(nn.Module):
    def __init__(self, shared_dim: int, p: float):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.shared_dim = shared_dim

    def forward(self, x: torch.tensor):
        out = None
        
        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = torch.ones(mask_shape, device=x.device)
        mask = self.dropout(mask)

        out = x * mask

        return out

class DropoutRowwise(SharedDropout):
    def __init__(self, p: float):
        super().__init__(shared_dim=-2, p=p)

class DropoutColumnwise(SharedDropout):
    def __init__(self, p: float):
        super().__init__(shared_dim=-3, p=p)
