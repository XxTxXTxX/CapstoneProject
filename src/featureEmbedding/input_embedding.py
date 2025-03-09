import torch
from torch import nn

class InputEmbedder(nn.Module):
    """
    Implements Algorithm 3 and Algorithm 4.
    """

    def __init__(self, c_m, c_z, tf_dim, msa_feat_dim=49, vbins=32):
        """
        Initializes the InputEmbedder.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation.
            tf_dim (int): Embedding dimension of target_feat.
            msa_feat_dim (int, optional): Embedding dimension of the MSA feature. 
                Defaults to 49.
            vbins (int, optional): Determines the bins for relpos as 
                (-vbins, -vbins+1,...,vbins). Defaults to 32.
        """
        super().__init__()
        self.tf_dim = tf_dim
        self.vbins = vbins
        
        # Initialize the modules linear_tf_z_i, linear_tf_z_j,linear_tf_m, linear_msa_m (from Algorithm 3) and linear_rel_pos (from Algorithm 4).

        # Note the difference between the MSA feature (as created during feature extraction) and the MSA representation m that is used throughout the Evoformer.
 
        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.linear_msa_m = nn.Linear(msa_feat_dim, c_m)
        self.linear_relpos = nn.Linear(2*vbins+1, c_z)

    def relpos(self, residue_index):
        """
        Implements Algorithm 4.

        Args:
            residue_index (torch.tensor): Index of the residue in the original amino
                acid sequence. In this context, this is simply [0,... N_res-1].

        Returns:
            tuple: Tuple consisting of the embedded MSA representation m and 
                pair representation z.
        """

        out = None
        dtype = self.linear_relpos.weight.dtype

        # Implement Algorithm 4. Since the residue index is just a number, we can directly use the shifted d_ij as class labels.                #
        # You can follow these steps:
        #   * Cast residue_index to long.                                        #
        #   * unsqueeze residue_index accordingly to calculate the outer         #
        #      difference d_ij.                                                  #
        #   * use torch.clamp to clamp d_ij between -self.vbins and self.vbins.  #
        #   * offset the clamped d_ij by self.vbins, so that it is in the range  #
        #      [0, 2*vbins] instead of [-vbins, vbins].                          #
        #   * use nn.functional.one_hot to convert the class labels into         #
        #      one-hot encodings.                                                #
        #   * use the linear module to create the output embedding.
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
        """
        Rotate the C_z embedding dimension in the input of shape (N_res, C_z)
            - rotation happens before outer sum between "a" and "b"
        returns rotated tensor of shape (N_res, C_z)
        """
        N_res, C_z = input.shape
        assert C_z % 3 == 0, "C_z must be divisible by 3"

        # Normalize pH and temperature
        pH_norm = torch.tensor(self.normalize(pH, pH_range[0], pH_range[1]), dtype = torch.float32)
        temp_norm = torch.tensor(self.normalize(temp, temp_range[0], temp_range[1]), dtype = torch.float32)

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
        ], dtype = torch.float32)

        # Rotation matrix for temperature (xz-plane --> y, "pointing at you")
        R_y = torch.tensor([
            [torch.cos(theta_temp), 0,   torch.sin(theta_temp)],
            [0,                   1,                        0],
            [-torch.sin(theta_temp), 0, torch.cos(theta_temp)]
        ], dtype = torch.float32)

        # 3, 3
        combined_rotation = R_y @ R_z
        rotated_tensor = torch.matmul(combined_rotation, tensor_reshaped.transpose(1,2))

        # transpose back to (N_res, num_split, 3), then to (N_res, C_z)
        final_tensor = rotated_tensor.transpose(1,2).reshape(N_res, C_z)
        print(final_tensor)
        return final_tensor



    def forward(self, batch):
        """
        Implements the forward pass for Algorithm 3.

        Args:
            batch (dict): Feature dictionary with the following entries:
                * msa_feat: Initial MSA feature of shape (*, N_seq, N_res, msa_feat_dim).
                * target_feat: Target feature of shape (*, N_res, tf_dim).
                * residue_index: Residue index of shape (*, N_res)

        Returns:
            tuple: Tuple consisting of the MSA representation m and the pair representation z.
        """

        m = None
        z = None

        # N_clust, N_res, f_c = 49
        msa_feat = batch['msa_feat']
        # N_res, f = 21
        target_feat = batch['target_feat']
        # N_res, 
        residue_index = batch['residue_index']

        # Implement the forward pass for Algorithm 3. For the calculation of the outer sum in line 2, the embeddings a and b must be unsqueezed correctly to allow for broadcasting along the N_res dim.
        
        # Note: For batched use, target_feat must be unsqueezed after the computation of a and b and before the computation of m, to match the number of dimensions of msa_feat.

        # N_res, 21 -> N_res, C_z
        a = self.linear_tf_z_i(target_feat)
        print("a = ", a)
        # N_res, 21 -> N_res, C_z
        b = self.linear_tf_z_j(target_feat)
        print("b = ", b)
        
        # Rotate "a" and "b" before outer sum
        a_rotated = self.rotate_embeddings(a)
        b_rotated = self.rotate_embeddings(b)

        # (N_res, 1, C_z) + (1, N_res, C_z) -> (N_res, N_res, C_z)
        z = a.unsqueeze(-2) + b.unsqueeze(-3)
        # N_res, N_res, C_z
        z += self.relpos(residue_index) 
        
        #TODO: Add PH and temperature rotations
        
        # (1, N_res, 21)
        target_feat = target_feat.unsqueeze(-3)

        # (N_clust, N_res, C_m) + (1, N_res, C_m) -> (N_clust, N_res, C_m)
        m = self.linear_msa_m(msa_feat) + self.linear_tf_m(target_feat)

        return m, z