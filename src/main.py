import torch
from torch import nn
from featureEmbedding import input_embedding, extra_msa_stack
import time
from evoformer.evoformer import EvoformerStack

c_z = 129
c_m = 256
tf_dim = 21
f_e = 25  # extra msa dimension
c_e = 64

batch = torch.load(
    'src/preprocess/control_values/full_batch.pt', map_location='cpu')
#print(batch.keys())
# print(batch["msa_feat"].shape)
# print(batch["target_feat"].shape)

# Test

# residue -> relpos + target_feat -> linear + linear to outer sum -> pair_representation (N_res, N_res, C_z = 128)
embedder = input_embedding.InputEmbedder(c_m, c_z, tf_dim)
m, z = embedder.forward(batch)
print(m.shape, z.shape)
extra_embedder = extra_msa_stack.ExtraMsaEmbedder(f_e, c_e)
extra_msa_representation = extra_embedder.forward(batch)[:1, :, :]
extra_embedder_block = extra_msa_stack.ExtraMsaStack(c_e, c_z)
start = time.time()
z = extra_embedder_block.forward(extra_msa_representation, z)
end = time.time()
print(f"time: {end-start}")


evoformer_stack = EvoformerStack(c_m, c_z, num_blocks=1)
m, z, s = evoformer_stack.forward(m, z)
print(m.shape, z.shape, s.shape)

# msa_feat -> linear + target_feat -> linear -> tile -> MSA_representation (N_seq -> number of msa, N_res -> number of residues, C_m = 256)

#---------------------------------
import torch
from torch import nn
from featureEmbedding import input_embedding, extra_msa_stack
import time
from evoformer.evoformer import EvoformerStack
from structure_module.structure_module import StructureModule  # Import StructureModule

# Load batch
batch = torch.load(
    'src/preprocess/control_values/full_batch.pt', map_location='cpu')

# Define embedding dimensions
c_z, c_m, tf_dim, c_e, f_e = 129, 256, 21, 64, 25

# Create EvoformerStack
evoformer_stack = EvoformerStack(c_m, c_z, num_blocks=1)

# Embed inputs
embedder = input_embedding.InputEmbedder(c_m, c_z, tf_dim)
m, z = embedder.forward(batch)

# Extract amino acid sequence labels (F)
msa_aatype = batch["msa_feat"][0, :, :20]  # First sequence
F = torch.argmax(msa_aatype, dim=-1)  # Convert one-hot to index form

# Run Evoformer
m, z, s = evoformer_stack(m, z)

# Create and run Structure Module
structure_module = StructureModule(c_s=384, c_z=129)
structure_outputs = structure_module(s, z, F)

# Print results
print(f"Final atom positions: {structure_outputs['final_positions']}")
print(f"Position mask: {structure_outputs['position_mask']}")
print(f"Pseudo-beta positions: {structure_outputs['pseudo_beta_positions']}")


HETATM    1  C   ACE B   1      63.171  29.285  32.777  1.00 59.03           C  
HETATM    2  O   ACE B   1      62.875  30.425  33.111  1.00 59.72           O  
HETATM    3  CH3 ACE B   1      62.795  28.107  33.603  1.00 58.32           C  
ATOM      4  N   SER B   2      64.065  33.046  34.490  1.00 54.47           N  

C: [1, 0, 0, ....]
O: [0, 1, 0 ,,  ... ]
CH3: [0, 0, 1, ...]
[1, 1, 1, 0000]