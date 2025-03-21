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
from training_data_preprocess import process_files

# ---------------- Preprocess data --------------
input_dirs = ['model/input_seqs', 'model/msa_raw']
for dir_path in input_dirs:
        print(f"\nProcessing files in {dir_path}")
        process_files(dir_path)

# -------------------- DEVICE SETUP --------------------
device = torch.device("cuda")
print(f"Using device: {device}")


# -------------------- CUSTOM MASKED MSE LOSS --------------------
class MaskedMSELoss(nn.Module):
    """
    Custom loss function that computes MSE loss only for non-masked positions.
    """
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')  # Compute loss per element

    def forward(self, pred, target):
        """
        Args:
            pred: (Nres, 37, 3) predicted coordinates
            target: (Nres, 37, 3) ground truth coordinates
        Returns:
            masked loss: Mean loss computed only for valid (non-masked) coordinates
        """
        #print("pred shape = ", pred.shape)
        #print("Target shape = ", target.shape)
        mask = (target != 0).any(dim=-1)  # Mask where at least one xyz value ≠ 0
        loss = self.mse(pred, target)  # Compute per-element MSE loss
        masked_loss = loss * mask.unsqueeze(-1)  # Apply mask (broadcasted to match xyz dims)

        # Avoid dividing by zero when all values are masked
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        return masked_loss.sum() / mask.sum()  # Mean over non-masked values

# -------------------- LOAD LATEST CHECKPOINT --------------------
def load_latest_checkpoint(model, model_dir="./model/model_weights/"):
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

    temp_pH_vals = read_pH_temp_csv("model/pH_temp.csv")
    full_ds = ProcessDataset(temp_pH_vals)

    # Fix dataset splitting issue
    train_ds_size = int(0.8 * len(full_ds))
    val_ds_size = len(full_ds) - train_ds_size
    train_ds, val_ds = random_split(full_ds, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader



# -------------------- MODEL SETUP --------------------
model = ProteinStructureModel()
model.to(device)
model = load_latest_checkpoint(model)  # Load checkpoint if available
train_dataloader, val_dataloader = get_ds()




# -------------------- TRAINING LOOP --------------------
from tqdm import tqdm

# -------------------- TRAINING LOOP --------------------
def train(model, train_loader, val_loader, num_epochs=40, lr=1e-3, device=device):
    """
    Trains the model using the custom masked MSE loss with tqdm progress bars.

    Args:
        model: The protein structure model (expects input shape Nres, 37, 3)
        train_loader: Training data DataLoader
        val_loader: Validation data DataLoader
        num_epochs: Number of epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedMSELoss()  # Use custom loss function

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Training loop with tqdm progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch") as t:
            count = 0
            for batch in t:
                count += 1
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                coordinates = batch['coordinates']  # Ground truth (Nres, 37, 3)
                #print(batch["seq_name"])
                pred = model(batch)  # Model returns a dictionary
                
                # Extract the relevant tensors
                pred_coords = pred["final_positions"]
                
                target_coords = coordinates  # Already extracted
                # match shape (added code)
                if pred_coords.shape[1] < target_coords.shape[1]:
                    target_coords = target_coords[:, :pred_coords.shape[1], :, :]
                elif pred_coords.shape[1] > target_coords.shape[1]:
                    pred_coords = pred_coords[:, :target_coords.shape[1], :, :]
                loss = criterion(pred_coords, target_coords)  # Compute loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                t.set_postfix(loss=train_loss / (t.n + 1))  # Update tqdm progress bar
                
                del batch, pred, pred_coords, target_coords, loss  
                torch.cuda.empty_cache()
            print(count)
        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch") as t:
                for batch in t:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    coordinates = batch['coordinates']
                    pred = model(batch)

                    # Extract the relevant tensors
                    pred_coords = pred["final_positions"]
                    target_coords = coordinates  # Already extracted
                    loss = criterion(pred_coords, target_coords)
                    val_loss += loss.item()

                    t.set_postfix(loss=val_loss / (t.n + 1))  # Update tqdm progress bar

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

        # Save the model after each epoch
        model_save_path = f"./model/model_weights/model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved: {model_save_path}")



# -------------------- RUN TRAINING --------------------
# next time lr = 5e-5
train(model, train_dataloader, val_dataloader, num_epochs=40, lr=5e-5, device=device)
print("finished!!")
