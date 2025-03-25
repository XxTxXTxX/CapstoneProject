import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Union, Tuple, Iterable

from featureEmbedding.input_embedding import InputEmbedder
from evoformer.evoformer import EvoformerStack
from featureEmbedding.extra_msa_stack import ExtraMsaEmbedder, ExtraMsaStack
from structure_module.structure_module import StructureModule

class ModelArgs:
    def __init__(
            self, 
            c_z: int, # pair representation embed dim
            c_m: int, # MSA embed dim
            tf_dim: int, # target feature embed dim
            f_e: int,# extra msa dimension
            c_e: int,
            max_seq_len: int = None,
            **kwargs
            ):
        super().__init__()
        # Evoformer arguments
        self.c_m = c_m
        self.tf_dim = tf_dim
        self.f_e = f_e
        self.c_e = c_e
    


### Model

class ProteinStructureModel(nn.Module):
    def __init__(self, 
                 c_m=256,        
                 c_z=129,        
                 c_e=64,
                 f_e=25,
                 tf_dim=21,         
                 num_blocks=1): 
        super().__init__()
        
        self.input_embedder = InputEmbedder(c_m, c_z, tf_dim)
        self.extra_msa_Embedder = ExtraMsaEmbedder(f_e, c_e)
        self.extra_msa_stack = ExtraMsaStack(c_e, c_z)
        self.evoformer_stack = EvoformerStack(c_m, c_z, num_blocks)
        self.structure_module = StructureModule(c_s=384, c_z=c_z)

    def forward(self, batch):
        # 1
        m, z = self.input_embedder.forward(batch)

        #print("Input embedder section: \n")
        #print("m's shape: ", m.shape)
        #print("z's shape: ", z.shape)

        extra_msa_representation = self.extra_msa_Embedder.forward(batch)
        #print("Extra msa representation = ", extra_msa_representation.shape)

        z = self.extra_msa_stack(extra_msa_representation, z)
        #print("Extra msa stack = ", extra_msa_representation.shape)
        
        # 2
        m, z, s = self.evoformer_stack(m, z)
        #print("Evoformer stack = ", m.shape, z.shape, s.shape)

        # 3
        #msa_aatype = batch['msa_feat'][0, :, :20]
        msa_aatype = batch['target_feat']
        F = torch.argmax(msa_aatype, dim=-1)
        # print(f"Structure Module F.shape: {F.shape}, Unique: {torch.unique(F)}")
        F = torch.clamp(F, min=0, max=19)
        #print("F = ", F.shape)
        output = self.structure_module(s, z, F)
        #print("Structure module = ", output["final_positions"].shape)
        
        return output



#
