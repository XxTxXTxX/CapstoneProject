from pathlib import Path
import sys
MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from .dataset_inference import ProcessDataset
from .model import ProteinStructureModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import torch.optim as optim
import random
import numpy as np
import os
from tqdm import tqdm
from training_data_preprocess import process_files

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

# -------------------- CSV LOADING --------------------
def read_pH_temp_csv():
    """
    Reads a CSV file containing pH and temperature values for each PDB ID.
    Returns:
        dict: {pdb_id: [pH, temperature]}
    """
    absolute_path = os.path.join(MODEL_DIR, "pH_temp_inference.csv")
    data_dict = {}
    with open(absolute_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            pdb_id = row[0]
            if row[1] == "Error":
                row[1] = 7 # Default ph Value
            ph = float(row[1])
            if row[2] == "Error":
                row[2] = 277 # Default temp value
            temp = float(row[2])
            data_dict[pdb_id] = [ph, temp]
    return data_dict


# -------------------- DATASET LOADING --------------------
def collate_fn(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    collated = {}
    for key in batch[0].keys():
        samples = [item[key] for item in batch]
        if isinstance(samples[0], torch.Tensor):
            try:
                collated[key] = torch.stack(samples).to(device)
            except RuntimeError:
                collated[key] = nn.utils.rnn.pad_sequence(samples, batch_first=True).to(device)
        else:
            collated[key] = samples 
    return collated

def get_ds(seed = 43):
    """
    Creates train and validation DataLoaders from ProcessDataset.
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    temp_pH_vals = read_pH_temp_csv()
    inference_ds = ProcessDataset(temp_pH_vals)

    inference_dataloader = DataLoader(inference_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return inference_dataloader


def run_inference(sequence, device=torch.device("cpu")):
    model = ProteinStructureModel().to(device)
    model = load_latest_checkpoint(model)
    model.eval()

    with torch.no_grad():
        pred = model({"sequence": sequence})
    
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

    pred_coords = pred_coords.squeeze(0) * 3
    pred_mask = pred_mask.squeeze(0)

    with open(output_path, "w") as f:
        atom_index = 1
        for res_idx, res_name in enumerate(sequence):
            res_name_3 = AA_1_to_3.get(res_name.upper(), 'UNK')
            for atom_idx, atom_name in enumerate(ATOM_TYPES):
                if not pred_mask[res_idx, atom_idx]:
                    continue
                x, y, z = pred_coords[res_idx, atom_idx].tolist()
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