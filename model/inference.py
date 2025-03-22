from dataset_inference import ProcessDataset
from model import ProteinStructureModel
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

# ---------------- Preprocess data --------------
input_dirs = ['model/input_seqs_inference', 'model/msa_raw_inference']
for dir_path in input_dirs:
        print(f"\nProcessing files in {dir_path}")
        process_files(dir_path)

# -------------------- DEVICE SETUP --------------------
device = torch.device("cpu")
print(f"Using device: {device}")


# -------------------- LOAD LATEST CHECKPOINT --------------------
def load_latest_checkpoint(model, model_dir="../model_weights/"):
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
def read_pH_temp_csv(file_path):
    """
    Reads a CSV file containing pH and temperature values for each PDB ID.
    Returns:
        dict: {pdb_id: [pH, temperature]}
    """
    data_dict = {}
    with open(file_path, mode="r") as file:
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

    temp_pH_vals = read_pH_temp_csv("model/pH_temp_inference.csv")
    inference_ds = ProcessDataset(temp_pH_vals)

    inference_dataloader = DataLoader(inference_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return inference_dataloader


inference_dataloader = get_ds()
model = ProteinStructureModel()
model.to(device)
model = load_latest_checkpoint(model)


def run_inference(model, inference_dataloader, device=device):
    model = ProteinStructureModel().to(device)
    model = load_latest_checkpoint(model)
    print("here")
    model.eval()
    for batch in inference_dataloader:
        pred = model(batch)
    
    pred_coords = pred["final_positions"]
    print(f"Predicted Coordinates Shape: {pred_coords.shape}")

    return pred_coords

result = run_inference(model, inference_dataloader, device)
print(result.shape)

def save_pdb(pred_coords, sequence, output_file="model/predict_pdb/predicted_structure_1914.pdb"):
    """
    Saves the predicted (Nres, 37, 3) tensor as a PDB file for visualization.
    """
    ATOM_TYPES = [
        "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG",
        "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
        "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "CZ", "CZ2", "CZ3", "NZ", "OXT", "OH", "TYR_OH"
    ]
    pred_coords = pred_coords.squeeze(0)  # Now shape is [232, 37, 3]
    with open(output_file, "w") as f:
        atom_index = 1  # PDB atom numbering starts at 1
        for res_idx, res_name in enumerate(sequence):
            for atom_idx, atom_name in enumerate(ATOM_TYPES):
                x, y, z = pred_coords[res_idx, atom_idx].tolist()  

                # Skip missing atoms (zero tensors)
                if x == 0.0 and y == 0.0 and z == 0.0:
                    continue  

                pdb_line = (
                    f"ATOM  {atom_index:5d} {atom_name:<4} {res_name:>3} A "
                    f"{res_idx+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>1}\n"
                )
                f.write(pdb_line)
                atom_index += 1

        f.write("TER\nEND\n")  # End of PDB
    print(f"PDB file saved: {output_file}")


# -------------------- RUN INFERENCE EXAMPLE --------------------
sequence = "MASMTGGQQMGRIPGNSPRMVLLESEQFLTELTRLFQKCRSSGSVFITLKKYDGRTKPIPRKSSVEGLEPAENKCLLRATDGKRKISTVVSSKEVNKFQMAYSNLLRANMDGLKKRDKKNKSKKSKPAQGGEQKLISEEDDSAGSPMPQFQTWEEFSRAAEKLYLADPMKVRVVLKYRHVDGNLCIKVTDDLVCLVYRTDQAQDVKKIEKFHSQLMRLMVAKESRNVTMETE"
save_pdb(result, sequence, "predicted_structure.pdb")