import torch
from torch import nn

from structure_module.ipa import InvariantPointAttention
from geometry.geometry import compute_all_atom_coordinates, assemble_4x4_transform, quat_to_3x3_rotation
from geometry.geometry import residue_constants
    

class StructureModuleTransition(nn.Module):
    def __init__(self, c_s):
        super().__init__()
        self.c_s = c_s

        self.linear_1 = nn.Linear(c_s, c_s)
        self.linear_2 = nn.Linear(c_s, c_s)
        self.linear_3 = nn.Linear(c_s, c_s)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(c_s)
        self.relu = nn.ReLU()

    def forward(self, s):

        s = s + self.linear_3(self.relu(self.linear_2(self.relu(self.linear_1(s)))))
        s = self.layer_norm(self.dropout(s))

        return s

class BackboneUpdate(nn.Module):
    def __init__(self, c_s):
        super().__init__()

        self.linear = nn.Linear(c_s, 6)

    def forward(self, s):

        T = None
        group = self.linear(s)
        quat = torch.cat((torch.ones(group.shape[:-1]+(1,), device=group.device), group[...,:3]), dim=-1)
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        t = group[..., 3:]

        # Explicit formula from Algorithm 22:
        # a, b, c, d = torch.unbind(quat, dim=-1)
        # R = [
        #     [a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
        #     [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
        #     [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]
        # ]
        # R = [torch.stack(vals, dim=-1) for vals in R]
        # R = torch.stack(R, dim=-2)

        R = quat_to_3x3_rotation(quat)
        T = assemble_4x4_transform(R,  t)
        return T

class AngleResNetLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.linear_1 = nn.Linear(c, c)
        self.linear_2 = nn.Linear(c, c)
        self.relu = nn.ReLU()

    def forward(self, a):
        a = a + self.linear_2(self.relu(self.linear_1(self.relu(a))))

        return a


class AngleResNet(nn.Module):
    
    def __init__(self, c_s, c, n_torsion_angles=7):
        super().__init__()
        self.n_torsion_angles = n_torsion_angles
        self.linear_in = nn.Linear(c_s, c)
        self.linear_initial = nn.Linear(c_s, c)
        self.layers = nn.ModuleList([AngleResNetLayer(c) for _ in range(2)])
        self.linear_out = nn.Linear(c, 2*n_torsion_angles)
        self.relu = nn.ReLU()


    def forward(self, s, s_initial):
        alpha = None
        # ReLUs absent in supplementary methods
        s = self.relu(s)
        s_initial = self.relu(s_initial)
        a = self.linear_in(s) + self.linear_initial(s_initial)
        for layer in self.layers:
            a = layer(a)
        alpha = self.linear_out(self.relu(a))
        alpha_shape = alpha.shape[:-1] + (self.n_torsion_angles, 2)
        alpha = alpha.view(alpha_shape)

        return alpha



class StructureModule(nn.Module):
    def __init__(self, c_s, c_z, n_layer=8, c=128):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.n_layer = n_layer
        # self.scale = nn.Parameter(torch.tensor(15.0))

        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_in = nn.Linear(c_s, c_s)

        self.layer_norm_ipa = nn.LayerNorm(c_s)
        self.dropout_s = nn.Dropout(0.1)
        self.ipa = InvariantPointAttention(c_s, c_z)
        self.transition = StructureModuleTransition(c_s)
        self.bb_update = BackboneUpdate(c_s)
        self.angle_resnet = AngleResNet(c_s, c)

    def process_outputs(self, T, alpha, F):
        final_positions, position_mask, pseudo_beta_positions = None, None, None

        scaled_T = T.clone()
        scaled_T[..., :3, 3] *= 10 #Modifed self.scale
        final_positions, position_mask = compute_all_atom_coordinates(scaled_T, alpha, F)

        c_beta_ind = residue_constants.atom_types.index('CB')
        c_alpha_ind = residue_constants.atom_types.index('CA')
        glycine_ind = residue_constants.restypes.index('G')
        pseudo_beta_positions = final_positions[..., c_beta_ind, :]
        alpha_positions = final_positions[..., c_alpha_ind, :]
        pseudo_beta_positions[F==glycine_ind] = alpha_positions[F==glycine_ind]

        return final_positions, position_mask, pseudo_beta_positions
        

    def forward(self, s, z, F):
        N_res = z.shape[-2]
        batch_dim = s.shape[:-2]
        outputs = {'angles': [], 'frames': []}
        device = s.device
        dtype = s.dtype

        # print(f"Structure Module F.shape: {F.shape}, Unique: {torch.unique(F)}")

        s_initial = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        s = self.linear_in(s_initial)
        T = torch.eye(4, device=device, dtype=dtype).broadcast_to(batch_dim+(N_res, 4, 4))

        for _ in range(self.n_layer):
            s = s + self.ipa(s, z, T)
            s = self.layer_norm_ipa(self.dropout_s(s))
            s = self.transition(s)
            T = T @ self.bb_update(s)

            alpha = self.angle_resnet(s, s_initial)
            outputs['angles'].append(alpha)
            outputs['frames'].append(T)

        outputs['angles'] = torch.stack(outputs['angles'], dim=-4)
        outputs['frames'] = torch.stack(outputs['frames'], dim=-4)

        final_positions, position_mask, pseudo_beta_positions = self.process_outputs(T, alpha, F)
        outputs['final_positions'] = final_positions
        outputs['position_mask'] = position_mask
        outputs['pseudo_beta_positions'] = pseudo_beta_positions

        return outputs
            

