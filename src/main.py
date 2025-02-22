import torch
from torch import nn
from featureEmbedding import input_embedding, extra_msa_stack

c_z = 128
c_m  = 256
tf_dim = 21
f_e = 25 # extra msa dimension
c_e = 64

batch = torch.load('src/preprocess/control_values/full_batch.pt', map_location='cpu')
print(batch.keys())
# print(batch["msa_feat"].shape)
# print(batch["target_feat"].shape)


# Test

# extra_msa -> Linear -> extra_MSA_representation ()


# residue -> relpos + target_feat -> linear + linear to outer sum -> pair_representation (N_res, N_res, C_z = 128)
# embedder = input_embedding.InputEmbedder(c_m, c_z, tf_dim)
# m, z = embedder.forward(batch)
# # print(m.shape)
# # print(z.shape)
# extra_embedder = extra_msa_stack.ExtraMsaEmbedder(f_e, c_e)
# extra_msa_representation = extra_embedder.forward(batch)
# # print(extra_embedder.forward(batch).shape)

# extra_embedder_block = extra_msa_stack.ExtraMsaBlock(c_e, c_z)
# stack = extra_embedder_block.forward(extra_msa_representation, z)
# print(stack.shape)

# msa_feat -> linear + target_feat -> linear -> tile -> MSA_representation (N_seq -> number of msa, N_res -> number of residues, C_m = 256)

