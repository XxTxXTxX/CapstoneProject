#--------------------------- train.py
from dataset import ProcessDataset
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

# -------------------- DEVICE SETUP --------------------
device = torch.device("cuda")
print(f"Using device: {device}")


# -------------------- CUSTOM MASKED MSE LOSS --------------------
"""
class MaskedMSELoss(nn.Module):
    def __init__(self, mask_penalty=1.0):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')  # Compute loss per element
        self.mask_penalty = mask_penalty

    
    def forward(self, pred, target, mask):
        target_mask = (target != 0).any(dim=-1)  # (batch, Nres, 37)
        
        # uclidean distance loss
        # coord_loss = torch.sum((pred - target) ** 2, dim=-1)  # (batch, Nres, 37)
        coord_loss = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1) + 1e-8) 
        # total_loss = torch.zeros_like(coord_loss)
        

        
        # pred mask = 1 but target is 0, wrong mask, penalty
        case1 = mask & ~target_mask
        # total_loss[case1] = self.mask_penalty * torch.ones_like(total_loss)[case1]
        total_loss[case1] = self.mask_penalty * coord_loss[case1].mean()
        
        # Correct mask and target is 0, no penalty
        case2 = ~mask & ~target_mask
        total_loss[case2] = 0.0
        
        # Both pred and target are not 0, calculate coord loss by uclidean distance
        case3 = mask & target_mask
        total_loss[case3] = coord_loss[case3]
        
        # pred mask = 0 but target coord != 0, penalty on both
        case4 = ~mask & target_mask
        total_loss[case4] = coord_loss[case4] + self.mask_penalty

        total_loss = (coord_loss[case3].mean() + self.mask_penalty * (case1.sum() + case4.sum())) / (valid_positions + 1e-8)
        
        # Average loss
        valid_positions = (case1 | case3 | case4).sum()
        if valid_positions == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        return {
            'total_loss': total_loss.sum() / (valid_positions + 1e-8),
            'coord_loss': coord_loss[case3].mean() if case3.sum() > 0 else torch.tensor(0.0),
            'mask_penalty': (total_loss[case1].sum() + total_loss[case4].sum()) / (case1.sum() + case4.sum() + 1e-8)
        }
"""

class MaskedMSELoss(nn.Module):
    """
    Computes coordinate loss only for visible atoms,
    and enforces realistic backbone geometry (e.g., CA–CA distance).
    """
    def __init__(self, mask_penalty=1.0, ca_bond_penalty=40.0, backbone_weight=8.0, expected_ca_dist=3.8):
        super(MaskedMSELoss, self).__init__()
        self.mask_penalty = mask_penalty
        self.ca_bond_penalty = ca_bond_penalty
        self.backbone_weight = backbone_weight
        self.expected_ca_dist = expected_ca_dist


    def forward(self, pred, target, mask):
        
        target_mask = (target != 0).any(dim=-1)  # (B, N, 37)
        coord_loss = torch.sum((pred - target) ** 2, dim=-1)  # (B, N, 37)
        total_loss = torch.zeros_like(coord_loss)

        # Penalty: predicted present but actually missing
        case1 = mask & ~target_mask
        total_loss[case1] = self.mask_penalty

        # No prediction and target also missing — no penalty
        case2 = ~mask & ~target_mask
        total_loss[case2] = 0.0

        # Correct prediction (present in both)
        case3 = mask & target_mask
        total_loss[case3] = coord_loss[case3]

        # Missed atom: target present but predicted missing
        case4 = ~mask & target_mask
        total_loss[case4] = coord_loss[case4] + self.mask_penalty

        # === Boost loss for backbone atoms ===
        backbone_indices = [0, 1, 2]  # N, CA, C
        backbone_mask = torch.zeros_like(mask)
        backbone_mask[..., backbone_indices] = 1
        backbone_mask = backbone_mask.bool() & case3  # apply only where atom is visible

        # Multiply coord loss of backbone atoms
        total_loss[backbone_mask] *= self.backbone_weight

        # === CA–CA bond distance loss ===
        # Extract CA coordinates (index 1)
        pred_ca = pred[:, :, 1, :]  # (B, N, 3)
        ca_dists = torch.norm(pred_ca[:, 1:] - pred_ca[:, :-1], dim=-1)  # (B, N-1)
        ca_bond_loss = ((ca_dists - self.expected_ca_dist) ** 2).mean()
        
        # Final loss aggregation
        valid_positions = (case1 | case3 | case4).sum()
        if valid_positions == 0:
            return {
                'total_loss': torch.tensor(0.0, requires_grad=True, device=pred.device),
                'coord_loss': torch.tensor(0.0),
                'mask_penalty': torch.tensor(0.0),
                'ca_bond_loss': torch.tensor(0.0),
            }

        return {
            #'total_loss': (total_loss.sum() / valid_positions * 0.1) + self.ca_bond_penalty * ca_bond_loss,
            'total_loss': self.ca_bond_penalty * ca_bond_loss,
            'coord_loss': coord_loss[case3].mean() if case3.sum() > 0 else torch.tensor(0.0),
            'mask_penalty': (total_loss[case1].sum() + total_loss[case4].sum()) / (case1.sum() + case4.sum() + 1e-8),
            'ca_bond_loss': ca_bond_loss
        }

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

    model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device), strict = False)
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


def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:  
            return None  
        collated = {}
        for key in batch[0].keys():
            samples = [item[key] for item in batch]
            if isinstance(samples[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(samples) 
                except RuntimeError:
                    collated[key] = nn.utils.rnn.pad_sequence(samples, batch_first=True)
            else:
                collated[key] = samples
        return collated

def get_ds():
    temp_pH_vals = read_pH_temp_csv("model/pH_temp.csv")
    full_ds = ProcessDataset(temp_pH_vals) 

    train_ds_size = int(0.85 * len(full_ds))
    val_ds_size = len(full_ds) - train_ds_size
    train_ds, val_ds = random_split(full_ds, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(
        train_ds, batch_size=1, shuffle=True, pin_memory=True,
        collate_fn=collate_fn, drop_last=True  
    )

    val_dataloader = DataLoader(
        val_ds, batch_size=1, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, drop_last=True
    )

    return train_dataloader, val_dataloader

# -------------------- MODEL SETUP --------------------
model = ProteinStructureModel()
model.to(device)
print("Model loading....")
model = load_latest_checkpoint(model)  # Load checkpoint if available
print("Model loading finished....")
print("Data loading...")
train_dataloader, val_dataloader = get_ds()
print("Data loading finished...")




# -------------------- TRAINING LOOP --------------------
from tqdm import tqdm

# -------------------- TRAINING LOOP --------------------
def train(model, train_loader, val_loader, num_epochs=20, lr=1e-3, device=device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedMSELoss()  # Use custom loss function

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0.0,
            'coord': 0.0,
            'mask_penalty': 0.0,
            'ca_bond': 0.0 # added CA-CA bond distance loss
        }

        # Training loop with tqdm progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch") as t:
            count = 0
            for batch in t:
                if batch == None:
                    continue
                count += 1
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                coordinates = batch['coordinates']  # Ground truth (Nres, 37, 3)
                print(batch["seq_name"])
                pred = model(batch)  # Model returns a dictionary
                
                # Extract the relevant tensors
                pred_coords = pred["final_positions"]

                target_coords = coordinates  # Already extracted
                #print(f"label: {target_coords.shape}")
                #print(f"prediction: {pred_coords.shape}")
                # match shape (added code)
                if pred_coords.shape[1] < target_coords.shape[1]:
                    target_coords = target_coords[:, :pred_coords.shape[1], :, :]
                elif pred_coords.shape[1] > target_coords.shape[1]:
                    pred_coords = pred_coords[:, :target_coords.shape[1], :, :]

                loss_dict = criterion(pred_coords, target_coords, pred['position_mask'])  # Compute loss
                loss = loss_dict['total_loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses['total'] += loss_dict['total_loss'].item()
                epoch_losses['coord'] += loss_dict['coord_loss'].item()
                # CA bond loss added
                epoch_losses['ca_bond'] += loss_dict['ca_bond_loss'].item()
                epoch_losses['mask_penalty'] += loss_dict['mask_penalty'].item()
                t.set_postfix({
                    'total_loss': epoch_losses['total'] / count,
                    'coord_loss': epoch_losses['coord'] / count,
                    'mask_loss': epoch_losses['mask_penalty'] / count,
                    # CA bond loss added
                    'CAbd': epoch_losses['ca_bond'] / count
                })  # Update tqdm progress bar
                
                del batch, pred, pred_coords, target_coords, loss  
                torch.cuda.empty_cache()
            print(count)
        # Validation
        model.eval()
        val_losses = {
            'total': 0.0,
            'coord': 0.0,
            'mask_penalty': 0.0,
            'ca_bond': 0.0
        }

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch") as t:
                for batch in t:
                    if batch == None:
                        continue
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    coordinates = batch['coordinates']
                    pred = model(batch)

                    # Extract the relevant tensors
                    pred_coords = pred["final_positions"]
                    target_coords = coordinates  # Already extracted
                    if pred_coords.shape[1] < coordinates.shape[1]:
                        coordinates = coordinates[:, :pred_coords.shape[1], :, :]
                    elif pred_coords.shape[1] > coordinates.shape[1]:
                        pred_coords = pred_coords[:, :coordinates.shape[1], :, :]
                    loss_dict = criterion(pred_coords, coordinates, pred['position_mask'])
                    val_losses['total'] += loss_dict['total_loss'].item()
                    val_losses['coord'] += loss_dict['coord_loss'].item()
                    # added CA loss
                    val_losses['ca_bond'] += loss_dict['ca_bond_loss'].item()
                    val_losses['mask_penalty'] += loss_dict['mask_penalty'].item()

                    t.set_postfix({
                        'val_total': val_losses['total'] / (t.n + 1),
                        'val_coord': val_losses['coord'] / (t.n + 1),
                        'val_mask': val_losses['mask_penalty'] / (t.n + 1),
                        'val_ca_bond': val_losses['ca_bond'] / (t.n + 1)
                    })  # Update tqdm progress bar

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        print(f"Training - Total Loss: {epoch_losses['total']/count:.4f}, "
              f"Coord Loss: {epoch_losses['coord']/count:.4f}, "
              f"Mask Loss: {epoch_losses['mask_penalty']/count:.4f}, "
              f"CA Bond: {epoch_losses['ca_bond']/count:.4f}")
        print(f"Validation - Total Loss: {val_losses['total']/(t.n+1):.4f}, "
              f"Coord Loss: {val_losses['coord']/(t.n+1):.4f}, "
              f"Mask Loss: {val_losses['mask_penalty']/(t.n+1):.4f}, "
              f"CAbd: {val_losses['ca_bond']/(t.n+1):.4f}")

        # Save the model after each epoch
        model_save_path = f"../model_weights/model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved: {model_save_path}")



# -------------------- RUN TRAINING --------------------
# next time lr = 5e-5
train(model, train_dataloader, val_dataloader, num_epochs=10, lr=5e-7, device=device)
print("finished!!")



