import torch
from torch import nn
from evoformer.dropout import DropoutRowwise
from evoformer.msa_stack import MSARowAttentionWithPairBias, MSAColumnAttention, OuterProductMean, MSATransition
from evoformer.pair_stack import PairStack
from evoformer.rotaryEmbedding import RotaryEmbedding

# Equivariant Graph NeuralNet layer 
# (refining residue-level features in a geometry-aware way)
class EGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            # 2*Cs+1, Cs
            nn.Linear(2 * in_dim + 1, out_dim),
            nn.SiLU(),
            # Cs, Cs
            nn.Linear(out_dim, out_dim)
        )
        self.node_mlp = nn.Sequential(
            # 2*Cs, Cs
            nn.Linear(in_dim + out_dim, out_dim),
            nn.SiLU(),
            # Cs, Cs
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, coords):
        # x: Nres, Cs (node features)
        # edge_index: 2, num_edges (connections)
        # coords: Nres, 3 (3D coordinates for edge distance computation, edge features)
        
        # row = num_edges (source nodes), col = num_edges (target nodes)
        row, col = edge_index
        
        # num_edges, Cs
        x_i, x_j = x[row], x[col]
        # num_edges, 3
        coord_diff = coords[row] - coords[col]
        # num_edges, 1 
        radial = (coord_diff ** 2).sum(dim=-1, keepdim=True)
        # num_edges, 2*Cs+1
        edge_input = torch.cat([x_i, x_j, radial], dim=-1)
        # num_edges, Cs
        edge_feat = self.edge_mlp(edge_input)
        # Nres, Cs
        agg = torch.zeros_like(x)
        # Nres, Cs (add all edge features back into their source nodes)
        agg.index_add_(0, row, edge_feat)
        # Nres, 2*Cs
        node_input = torch.cat([x, agg], dim=-1)

        # Nres, Cs
        return self.node_mlp(node_input)


class EvoformerBlock(nn.Module):
    def __init__(self, c_m, c_z):
        super().__init__()

        self.dropout_rowwise_m = DropoutRowwise(p=0.15)
        self.msa_att_row = MSARowAttentionWithPairBias(c_m, c_z)
        self.msa_att_col = MSAColumnAttention(c_m)
        self.msa_transition = MSATransition(c_m)
        self.outer_product_mean = OuterProductMean(c_m, c_z)
        self.core = PairStack(c_z)

    def forward(self, m, z):

        m = m + self.dropout_rowwise_m(self.msa_att_row(m, z))
        m = m + self.msa_att_col(m)
        m = m + self.msa_transition(m)

        z = z + self.outer_product_mean(m)

        z = self.core(z)

        return m, z


class EvoformerStack(nn.Module):
    def __init__(self, c_m, c_z, num_blocks, c_s=384):
        super().__init__()
        self.blocks = nn.ModuleList([
            EvoformerBlock(c_m, c_z) for _ in range(num_blocks)
        ])
        self.linear = nn.Linear(c_m, c_s)

        self.coord_predictor = nn.Linear(c_s, 3)
        self.egnn = EGNNLayer(c_s, c_s)

    def forward(self, m, z):
        # m: batch, Nseq, Nres, Cm
        # z: batch, Nres, Nres, Cz

        for evo_block in self.blocks:
            m, z = evo_block(m, z)

        # Single representation extraction (1st row in MSA)
        # batch, Nres, Cs
        s = self.linear(m[..., 0, :, :])

        s_out = []
        # s_seq: (Nres, Cs), for single representation sequence
        for s_seq in s:
            # N: Nres
            N = s_seq.size(0)
            # Build graph: residues as nodes (bidirectional chain graph)
            # edge_index: 2, num_edges = 2*(Nres-1)
            edge_index = torch.tensor(
                [[i, i + 1] for i in range(N - 1)] +
                [[i + 1, i] for i in range(N - 1)],
                dtype=torch.long,
                device=s_seq.device
            ).t().contiguous()
            # Nres, 3
            coords = self.coord_predictor(s_seq)

            s_seq = self.egnn(s_seq, edge_index, coords)
            s_out.append(s_seq)

        s = torch.stack(s_out, dim=0)  # shape: [B, N_res, c_s]

        return m, z, s
