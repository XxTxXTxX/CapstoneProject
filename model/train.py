from dataset import ProcessDataset
from model import ProteinStructureModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import torch.optim as optim


# -------------------- DEVICE SETUP --------------------
device = torch.device("cpu")
print(f"Using device: {device}")
#device = torch.device("cpu")


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
        mask = (target != 0).any(dim=-1)  # Mask where at least one xyz value ≠ 0
        loss = self.mse(pred, target)  # Compute per-element MSE loss
        masked_loss = loss * mask.unsqueeze(-1)  # Apply mask (broadcasted to match xyz dims)

        # Avoid dividing by zero when all values are masked
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        return masked_loss.sum() / mask.sum()  # Mean over non-masked values


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
            ph = float(row[1])
            temp = float(row[2])
            data_dict[pdb_id] = [ph, temp]
    return data_dict


# -------------------- DATASET LOADING --------------------
def get_ds():
    """
    Creates train and validation DataLoaders from ProcessDataset.
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    temp_pH_vals = read_pH_temp_csv("model/pH_temp.csv")
    full_ds = ProcessDataset(temp_pH_vals)

    # Fix dataset splitting issue
    train_ds_size = int(0.8 * len(full_ds))
    val_ds_size = len(full_ds) - train_ds_size
    train_ds, val_ds = random_split(full_ds, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader


# -------------------- MODEL SETUP --------------------
model = ProteinStructureModel().to(device)
train_dataloader, val_dataloader = get_ds()


# -------------------- TRAINING LOOP --------------------
def train(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device=device):
    """
    Trains the model using the custom masked MSE loss.

    Args:
        model: The protein structure model (expects input shape Nres, 37, 3)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of epochs
        lr: Learning rate
        device: 'cuda', 'mps', or 'cpu'
    """
    model.to(device)  # Ensure model is on correct device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedMSELoss().to(device)  # Move loss function to device

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            if 'coordinates' not in batch or 'target_feat' not in batch:
                print("Warning: Missing keys in batch, skipping iteration!")
                continue

            # Move all tensors to MPS and convert to float32
            batch = {key: val.to(device, dtype=torch.float32) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}

            coordinates = batch['coordinates']  # Ground truth (Nres, 37, 3)
            pred_coords = model(batch)  # Model output (Nres, 37, 3)

            loss = criterion(pred_coords, coordinates)  # Compute loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if 'coordinates' not in batch:
                    continue

                batch = {key: val.to(device, dtype=torch.float32) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}

                coordinates = batch['coordinates']
                pred_coords = model(batch)
                loss = criterion(pred_coords, coordinates)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# -------------------- RUN TRAINING --------------------
train(model, train_dataloader, val_dataloader, num_epochs=20, lr=1e-4, device=device)