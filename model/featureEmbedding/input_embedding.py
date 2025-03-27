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
        # normalize value x to range [0,1] based on lower and upper bound
        return (x-min_val)/(max_val-min_val)

    def rotate_embeddings(
        self,
        input,
        pH: float = 7.7,
        temp: float = 280,
        pH_range = [0, 14],
        temp_range = [273, 300]
    ):
        input = input.squeeze(dim=0)
        device = input.device
        N_res, C_z = input.shape
        input.unsqueeze(dim=0)
        assert C_z % 3 == 0, "C_z must be divisible by 3"

        # Normalize pH and temperature
        pH_norm = torch.tensor(self.normalize(pH, pH_range[0], pH_range[1]), dtype = torch.float32, device=device)
        temp_norm = torch.tensor(self.normalize(temp, temp_range[0], temp_range[1]), dtype = torch.float32, device=device)

        # convert normalized values into rotation angles 
        theta_pH = pH_norm * (torch.pi/2)
        theta_temp = temp_norm * (torch.pi/2)

        num_splits = C_z // 3
        # N_res, num_splits, 3
        tensor_reshaped = input.view(N_res, num_splits, 3)

        # Rotation matrix for pH (xy-plane --> z, vertical rotation)
        R_z = torch.tensor([
            [torch.cos(theta_pH), -torch.sin(theta_pH), 0],
            [torch.sin(theta_pH), torch.cos(theta_pH),  0],
            [0                  , 0                   , 1]
        ], dtype = torch.float32, device=device)

        # Rotation matrix for temperature (xz-plane --> y, "pointing at you")
        R_y = torch.tensor([
            [torch.cos(theta_temp), 0,   torch.sin(theta_temp)],
            [0,                   1,                        0],
            [-torch.sin(theta_temp), 0, torch.cos(theta_temp)]
        ], dtype = torch.float32, device=device)

        # 3, 3
        combined_rotation = R_y @ R_z
        # (3, 3) @ (N_res, 3, num_splits) --> (N_res, 3, num_splits)
        rotated_tensor = torch.matmul(combined_rotation, tensor_reshaped.transpose(1,2))

        # transpose back to (N_res, num_split, 3), then to (N_res, C_z)
        final_tensor = rotated_tensor.transpose(1,2).reshape(N_res, C_z)
        return final_tensor



    def forward(self, batch):
        m = None
        z = None

        # N_clust, N_res, f_c = 49
        msa_feat = batch['msa_feat']
        # N_res, f = 21
        target_feat = batch['target_feat']
        # N_res, 
        residue_index = batch['residue_index']
        # N_res, 21 -> N_res, C_z
        a = self.linear_tf_z_i(target_feat)
        # N_res, 21 -> N_res, C_z
        b = self.linear_tf_z_j(target_feat)
        
        # Rotate "a" and "b" before outer sum
        a_rotated = self.rotate_embeddings(a)
        b_rotated = self.rotate_embeddings(b)

        # (N_res, 1, C_z) + (1, N_res, C_z) -> (N_res, N_res, C_z)
        z = a_rotated.unsqueeze(-2) + b_rotated.unsqueeze(-3)
        # N_res, N_res, C_z
        z = z + self.relpos(residue_index.squeeze(0))        
        
        # (1, N_res, 21)
        target_feat = target_feat.unsqueeze(-3)

        # (N_clust, N_res, C_m) + (1, N_res, C_m) -> (N_clust, N_res, C_m)
        m = self.linear_msa_m(msa_feat) + self.linear_tf_m(target_feat)
        z = z.unsqueeze(0)
        return m, z