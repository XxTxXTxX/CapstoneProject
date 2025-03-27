import torch
from torch import nn
# from tests.structure_module.residue_constants import rigid_group_atom_position_map, chi_angles_mask
from geometry.residue_constants import rigid_group_atom_position_map, chi_angles_mask,  chi_angles_chain
from geometry import residue_constants

def create_3x3_rotation(ex, ey):
    R = None
    ex = ex / torch.linalg.vector_norm(ex, dim=-1, keepdim=True)
    ey = ey - ex * torch.sum(ex*ey, dim=-1, keepdim=True)
    ey = ey / torch.linalg.vector_norm(ey, dim=-1, keepdim=True)
    ez = torch.linalg.cross(ex, ey, dim=-1)
    R = torch.stack((ex, ey, ez), dim=-1)

    return R

def quat_from_axis(phi, n):
    q = None

    a = torch.cos(phi/2).unsqueeze(-1)
    v = torch.sin(phi/2).unsqueeze(-1) * n
    q = torch.cat((a, v), dim=-1)

    return q

def quat_mul(q1, q2):

    a1 = q1[...,0:1] # a1 has shape (*, 1)
    v1 = q1[..., 1:] # v1 has shape (*, 3)
    a2 = q2[...,0:1] # a2 has shape (*, 1)
    v2 = q2[..., 1:] # v2 has shape (*, 3)

    q_out = None

    a_out = a1*a2 - torch.sum(v1*v2, dim=-1, keepdim=True)
    v_out = a1*v2 + a2*v1 + torch.linalg.cross(v1, v2, dim=-1)

    q_out = torch.cat((a_out, v_out), dim=-1)

    return q_out

def conjugate_quat(q):
    q_out = None

    q_out = q.clone()
    q_out[..., 1:] = -q_out[..., 1:]

    return q_out

def quat_vector_mul(q, v):
    batch_shape = v.shape[:-1]
    v_out = None

    zero_pad = torch.zeros(batch_shape+(1,), device=v.device, dtype=v.dtype)
    padded_v = torch.cat((zero_pad, v), dim=-1)

    q_out = quat_mul(q, quat_mul(padded_v, conjugate_quat(q)))
    v_out = q_out[...,1:]

    return v_out

def quat_to_3x3_rotation(q):

    R = None

    batch_shape = q.shape[:-1]
    eye = torch.eye(3, dtype=q.dtype, device=q.device)
    eye = eye.broadcast_to(batch_shape+(3,3))
    e1 = quat_vector_mul(q, eye[...,0])
    e2 = quat_vector_mul(q, eye[...,1])
    e3 = quat_vector_mul(q, eye[...,2])

    R = torch.stack((e1,e2,e3), dim=-1)
    
    return R

def assemble_4x4_transform(R, t):

    T = None
    batch_shape = t.shape[:-1]

    Rt = torch.cat((R, t[..., None]), dim=-1)
    pad = torch.zeros(batch_shape+(1, 4), device=t.device, dtype=t.dtype)
    pad[..., -1] = 1
    T = torch.cat((Rt, pad), dim=-2)
    return T

def warp_3d_point(T, x):
    x_warped = None
    device = x.device
    dtype = x.dtype
    pad = torch.ones(x.shape[:-1] + (1,), device=device, dtype=dtype)
    x_padded = torch.cat((x, pad), dim=-1)
    x_warped = torch.einsum('...ij,...j->...i', T, x_padded)
    x_warped = x_warped[..., :3]
    return x_warped
    
    

def create_4x4_transform(ex, ey, translation):
    T = None
    R = create_3x3_rotation(ex, ey)
    T = assemble_4x4_transform(R, translation)
    return T
    
def invert_4x4_transform(T):
    inv_T = None 

    R = T[...,:3,:3]
    t = T[...,:3,3]
    inv_R = R.transpose(-1, -2)
    inv_t = - torch.einsum('...ij,...j', inv_R, t)
    inv_T = assemble_4x4_transform(inv_R, inv_t)
    return inv_T

def makeRotX(phi):
    batch_shape = phi.shape[:-1]
    device = phi.device
    dtype = phi.dtype
    phi1, phi2 = torch.unbind(phi, dim=-1)
    T = None

    R = torch.zeros(batch_shape+(3,3), device=device, dtype=dtype)
    R[..., 0, 0] = 1
    R[..., 1, 1] = phi1
    R[..., 2, 1] = phi2
    R[..., 1, 2] = -phi2
    R[..., 2, 2] = phi1

    t = torch.zeros(batch_shape+(3,), device=device, dtype=dtype)
    T = assemble_4x4_transform(R, t)

    return T


def calculate_non_chi_transforms():
    non_chi_transforms = None

    backbone_group = torch.eye(4).broadcast_to(20, 4, 4)
    pre_omega_group = torch.eye(4).broadcast_to(20, 4, 4)
    phi_group = torch.zeros((20, 4, 4))
    psi_group = torch.zeros((20, 4, 4))

    for i, atom_positions in enumerate(rigid_group_atom_position_map.values()):
        ex_phi = atom_positions['N'] - atom_positions['CA']
        ey_phi = torch.tensor([1.0, 0.0, 0.0])
        aa_phi_group = create_4x4_transform(
            ex=ex_phi,
            ey=ey_phi,
            translation=atom_positions['N']
        )
        phi_group[i, ...] = aa_phi_group

        ex_psi = atom_positions['C'] - atom_positions['CA']
        ey_psi = atom_positions['CA'] - atom_positions['N']
        aa_psi_group = create_4x4_transform(
            ex=ex_psi,
            ey=ey_psi,
            translation=atom_positions['C']
        )
        psi_group[i, ...] = aa_psi_group
    
    non_chi_transforms = torch.stack(
        (backbone_group, pre_omega_group, phi_group, psi_group),
        dim=1
    )
    return non_chi_transforms

def calculate_chi_transforms():
    chi_transforms = None
    chi_transforms = torch.zeros(20, 4, 4, 4)

    for i, (aa, atom_positions) in enumerate(rigid_group_atom_position_map.items()):
        for j in range(4):
            if not chi_angles_mask[i][j]:
                chi_transforms[i,j] = torch.eye(4)
                continue

            next_atom = chi_angles_chain[aa][j]

            if j==0:
                ex = atom_positions[next_atom] - atom_positions['CA']
                ey = atom_positions['N'] - atom_positions['CA']
            else:
                ex = atom_positions[next_atom]
                ey = torch.tensor([-1.0,0.0,0.0])

            chi_transforms[i,j,...] = create_4x4_transform(
                ex=ex,
                ey=ey,
                translation=atom_positions[next_atom]
            )
    return chi_transforms

def precalculate_rigid_transforms():
    rigid_transforms = None
    non_chi_transforms = calculate_non_chi_transforms()
    chi_transforms = calculate_chi_transforms()

    rigid_transforms = torch.cat((non_chi_transforms, chi_transforms), dim=1)
    return rigid_transforms

def compute_global_transforms(T, alpha, F):
    global_transforms = None
    device = T.device
    dtype = T.dtype
    alpha = alpha / torch.linalg.vector_norm(alpha, dim=-1, keepdim=True)
    omega, phi, psi, chi1, chi2, chi3, chi4 = torch.unbind(alpha, dim=-2)
    all_rigid_transforms = precalculate_rigid_transforms().to(dtype=dtype, device=device)

    local_transforms = all_rigid_transforms[F]
    global_transforms = torch.zeros_like(local_transforms)

    global_transforms[..., 0, :, :] = T

    for i, ang in zip(range(1, 5), [omega, phi, psi, chi1]):
        global_transforms[..., i, :, :] = \
            T @ local_transforms[..., i, :, :] @ makeRotX(ang)

    for i, ang in zip(range(5, 8), [chi2, chi3, chi4]):
        global_transforms[..., i, :, :] = \
            global_transforms[..., i-1, :, :] @ local_transforms[..., i, :, :] @ makeRotX(ang)
    return global_transforms

def compute_all_atom_coordinates(T, alpha, F):
    global_positions, atom_mask = None, None
    device = T.device
    dtype = T.dtype
    global_transforms = compute_global_transforms(T, alpha, F)

    atom_local_positions = residue_constants.atom_local_positions.to(device=device,dtype=dtype)
    atom_frame_inds = residue_constants.atom_frame_inds.to(device=device)

    atom_local_positions = atom_local_positions[F]
    atom_frame_inds = atom_frame_inds[F]

    # Non-batched, array indexing:
    # seq_ind = torch.arange(atom_frame_inds.shape[0]).unsqueeze(-1)
    # seq_ind = seq_ind.broadcast_to(atom_frame_inds.shape)
    # atom_frames = global_transforms[seq_ind, atom_frame_inds]

    # Batched, torch.gather:
    dim_diff = global_transforms.ndim - atom_frame_inds.ndim
    atom_frame_inds = atom_frame_inds.reshape(atom_frame_inds.shape + (1,) * dim_diff)
    atom_frame_inds = atom_frame_inds.broadcast_to(atom_frame_inds.shape[:-dim_diff] + global_transforms.shape[-dim_diff:])
    atom_frames = torch.gather(global_transforms, dim=-3, index=atom_frame_inds)

    position_pad = torch.ones(atom_local_positions.shape[:-1]+(1,), device=device, dtype=dtype)
    padded_local_positions = torch.cat((atom_local_positions, position_pad), dim=-1)

    global_positions_padded = torch.einsum('...ijk,...ik->...ij', atom_frames, padded_local_positions)
    global_positions = global_positions_padded[...,:3]

    atom_mask = residue_constants.atom_mask.to(alpha.device)[F]
    return global_positions, atom_mask

