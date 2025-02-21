import torch
from torch import nn

c_z = 128
c_m  = 256
tf_dim = 21

batch = torch.load('preprocess/control_values/full_batch.pt', map_location='cpu')
# print(batch)
print(batch['target_feat'].shape)