from pathlib import Path
import sys
MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from .dataset_inference import ProcessDataset
from .model import ProteinStructureModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# -------------------- DEVICE SETUP --------------------
device = torch.device("cpu")
print(f"Using device: {device}")


# -------------------- LOAD LATEST CHECKPOINT --------------------
def load_latest_checkpoint(model, model_dir="./model_weights/"):
    os.makedirs(model_dir, exist_ok=True)  

    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not checkpoint_files:
        print("No checkpoint found, training from scratch.")
        return model  

    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  
    latest_checkpoint = checkpoint_files[-1]
    latest_checkpoint_path = os.path.join(model_dir, latest_checkpoint)

    model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {latest_checkpoint_path}")
    return model

def run_inference(sequence, pH=7.0, temperature=25.0, device=torch.device("cpu")):
    VALID_AA = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","X","-"]
    
    sequence = sequence.upper()
    invalid_chars = set(sequence) - set(VALID_AA)
    if invalid_chars:
        raise ValueError(
            f"Invalid amino acids found: {', '.join(invalid_chars)}. "
            f"Valid amino acids are: {', '.join(VALID_AA)}"
        )
    model = ProteinStructureModel().to(device)
    model = load_latest_checkpoint(model)
    dataset = ProcessDataset(sequence=sequence, pH=pH, temperature=temperature)
    if len(dataset) == 0:
        raise ValueError("Failed to process input sequence")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("dataloder = ", dataloader)
    model.eval()
    with torch.no_grad():
        for batch in dataloader: 
            pred = model(batch)
            
            pred_coords = pred["final_positions"]
        pred_mask = pred["position_mask"]
        # print(f"Pred Coords Min: {pred_coords.min()}, Max: {pred_coords.max()}, Mean: {pred_coords.mean()}")
        ca_coords = pred_coords[0,:,1]
        dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=-1)
        print("Avg CA-CA dist:", dists.mean().item())
        print(f"Predicted Coordinates Shape: {pred_coords.shape}")
        print(f"Predicted position mask: {pred_mask.shape}")
        print(pred_mask)

        return pred_coords, pred_mask
    
    raise RuntimeError("No predictions generated")


def save_pdb(pred_coords, pred_mask, sequence, output_file):
    MODEL_DIR = Path(__file__).parent
    output_path = os.path.join(MODEL_DIR, output_file)
    
    AA_1_to_3 = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }

    ATOM_TYPES = [
        "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG",
        "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
        "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "CZ", "CZ2", "CZ3", "NZ", "OXT", "OH", "TYR_OH"
    ]

    pred_coords = pred_coords.squeeze(0)
    pred_mask = pred_mask.squeeze(0)

    with open(output_path, "w") as f:
        atom_index = 1
        for res_idx, res_name in enumerate(sequence):
            res_name_3 = AA_1_to_3.get(res_name.upper(), 'UNK')
            for atom_idx, atom_name in enumerate(ATOM_TYPES):
                if not pred_mask[res_idx, atom_idx]:
                    continue
                x, y, z = pred_coords[res_idx, atom_idx].tolist()
                print([x, y, z])
                if x == 0.0 and y == 0.0 and z == 0.0:
                    continue
                pdb_line = (
                    f"ATOM  {atom_index:5d}  {atom_name:<4} {res_name_3:>3} A "
                    f"{res_idx+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>1}\n"
                )
                f.write(pdb_line)
                atom_index += 1
        f.write("TER\nEND\n")
    print(f"PDB file saved: {output_path}")



# -------------------- RUN INFERENCE EXAMPLE --------------------
# pred_coords, pred_mask = run_inference(model, inference_dataloader, device)
# sequence = "NLYQFKNMIKCTVPSRSWWDFADYGCYCGRGGSGTPVDDLDRCCQVHDNCYNEAEKISGCWPYFKTYSYECSQGTLTCKGDNNACAASVCDCDRLAAICFAGAPYNDNNYNIDLKARCQ"
# save_pdb(pred_coords, pred_mask, sequence, "output.pdb")