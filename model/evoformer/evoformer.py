import torch
from torch import nn
from evoformer.dropout import DropoutRowwise
from evoformer.msa_stack import MSARowAttentionWithPairBias, MSAColumnAttention, OuterProductMean, MSATransition
from evoformer.pair_stack import PairStack
from evoformer.rotaryEmbedding import RotaryEmbedding


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

        self.blocks = nn.ModuleList(
            [EvoformerBlock(c_m, c_z) for _ in range(num_blocks)])
        self.linear = nn.Linear(c_m, c_s)

    def forward(self, m, z):

        s = None

        i = 1
        for evo_block in self.blocks:
            #print(f"block nnumber:{i}\n")
            i += 1
            m, z = evo_block(m, z)

        s = self.linear(m[..., 0, :, :])
        #print(f"s:{s.shape}")

        return m, z, s
