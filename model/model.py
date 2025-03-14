import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Union, Tuple, Iterable

from featureEmbedding.input_embedding import InputEmbedder
from model.evoformer.evoformer import EvoformerStack
from model.featureEmbedding.extra_msa_stack import ExtraMsaEmbedder, ExtraMsaStack
from model.structure_module.structure_module import StructureModule

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
        # calculate m,z
        m, z = self.input_embedder.forward(batch)
        print("m = ", m.shape)
        print("Input embedder z = ", z.shape)
        extra_msa_representation = self.extra_msa_Embedder.forward(batch)[:1, :, :]
        print("Extra msa representation = ", extra_msa_representation.shape)
        z = self.extra_msa_stack(extra_msa_representation, z)
        print("Extra msa stack = ", extra_msa_representation.shape)
        m, z, s = self.evoformer_stack(m, z)
        print("Evoformer stack = ", m.shape, z.shape, s.shape)

        msa_aatype = batch['msa_feat'][0, :, :20]
        F = torch.argmax(msa_aatype, dim=-1)
        print("F = ", F.shape)
        output = self.structure_module(s, z, F)
        print("Structure module = ", output["final_positions"])
        
        return output



### Interact

def train():
    model = ProteinStructureModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 前向传播
            pred_positions = model(batch)
            
            # 计算损失
            loss = compute_structure_loss(pred_positions, batch['native_positions'])
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()