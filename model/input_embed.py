import torch
from torch import nn

class InputEmbedder(nn.Module):

    def __init__(self, c_m, c_z, tf_dim, msa_feat_dim=49, vbins=32):
        super().__init__()
        self.tf_dim = tf_dim
        self.vbins = vbins
        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.linear_msa_m = nn.Linear(msa_feat_dim, c_m)
        self.linear_relpos = nn.Linear(2*vbins+1, c_z)

    def relpos(self, residue_index):
        out = None
        dtype = self.linear_relpos.weight.dtype
        residue_index = residue_index.long()
        d = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        d = torch.clamp(d, -self.vbins, self.vbins) + self.vbins
        d_onehot = nn.functional.one_hot(d, num_classes=2*self.vbins+1).to(dtype=dtype)
        out = self.linear_relpos(d_onehot)

        return out

    def normalize(self, x, min_val, max_val):
        return (x-min_val)/(max_val-min_val)

    def rotate_embeddings(
        self,
        input,
        pH: float = 7.7,
        temp: float = 280,
        pH_range = [0, 14],
        temp_range = [273, 300]
    ):
        N_res, C_z = input.shape
        assert C_z % 3 == 0, "C_z must be divisible by 3"

        pH_norm = torch.tensor(self.normalize(pH, pH_range[0], pH_range[1]), dtype = torch.float32)
        temp_norm = torch.tensor(self.normalize(temp, temp_range[0], temp_range[1]), dtype = torch.float32)

        theta_pH = pH_norm * (torch.pi/2)
        theta_temp = temp_norm * (torch.pi/2)

        num_splits = C_z // 3
        tensor_reshaped = input.view(N_res, num_splits, 3)

        R_z = torch.tensor([
            [torch.cos(theta_pH), -torch.sin(theta_pH), 0],
            [torch.sin(theta_pH), torch.cos(theta_pH),  0],
            [0                  , 0                   , 1]
        ], dtype = torch.float32)

        R_y = torch.tensor([
            [torch.cos(theta_temp), 0,   torch.sin(theta_temp)],
            [0,                   1,                        0],
            [-torch.sin(theta_temp), 0, torch.cos(theta_temp)]
        ], dtype = torch.float32)

        combined_rotation = R_y @ R_z
        rotated_tensor = torch.matmul(combined_rotation, tensor_reshaped.transpose(1,2))

        final_tensor = rotated_tensor.transpose(1,2).reshape(N_res, C_z)
        print(final_tensor)
        return final_tensor



    def forward(self, batch):
        m = None
        z = None

        msa_feat = batch['msa_feat']
        target_feat = batch['target_feat']
        residue_index = batch['residue_index']

        a = self.linear_tf_z_i(target_feat)
        print("a = ", a)
        b = self.linear_tf_z_j(target_feat)
        print("b = ", b)
        
        a_rotated = self.rotate_embeddings(a)
        b_rotated = self.rotate_embeddings(b)

        z = a_rotated.unsqueeze(-2) + b_rotated.unsqueeze(-3)
        z += self.relpos(residue_index) 
        target_feat = target_feat.unsqueeze(-3)

        m = self.linear_msa_m(msa_feat) + self.linear_tf_m(target_feat)

        return m, z