import torch
from train import get_ds  # Ensure this function loads data correctly
from model import ProteinStructureModel

# Set device
device = torch.device("cpu")
print("here")


# Load dataset
train_dataloader, val_dataloader = get_ds()

# Get one batch
batch = next(iter(val_dataloader))
print(batch.keys())
for key, val in batch.items():
    if isinstance(val, torch.Tensor):
        batch[key] = val.to(device)

# Initialize the model
model = ProteinStructureModel().to(device)

# Forward pass with debugging
print("Starting forward pass...")
with torch.no_grad():
    print(f"seq_name: {batch["seq_name"]}")

    # Input embeddings
    m, z = model.input_embedder.forward(batch)
    print(f"Input embedder - m: {m.shape}, z: {z.shape}")

    # Extra MSA
    extra_msa_representation = model.extra_msa_Embedder.forward(batch)[:1, :, :]
    print(f"Extra MSA Representation: {extra_msa_representation.shape}")

    z = model.extra_msa_stack(extra_msa_representation, z)
    print(f"After Extra MSA Stack - z: {z.shape}")

    # Evoformer
    m, z, s = model.evoformer_stack(m, z)
    print(f"After Evoformer Stack - m: {m.shape}, z: {z.shape}, s: {s.shape}")

    # Prepare structure module inputs
    # msa_aatype = batch['msa_feat']
    # F = torch.argmax(msa_aatype, dim=-1)
    F = batch['target_feat'].argmax(dim=-1) - 1
    print(f"Structure module input F shape: {F.shape}")

    # Structure Module
    output = model.structure_module(s, z, F)
    print(f"Structure module output keys: {output.keys()}")
    
    final_positions = output["final_positions"]
    print(f"Final positions shape: {final_positions.shape}")

print("Forward pass completed!")