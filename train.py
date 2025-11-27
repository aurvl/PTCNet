import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure src is in path if running from root
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.data.data_lauder import prepare_and_save_datasets, get_dataloaders
from src.ptck_arch.arch import PTCNet, ModelA_NoInteractions, ModelB_WithInteractions

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    for batch in loader:
        # Unpack batch (handling the 4-tuple from collate: X, y, time_mask, feat_mask)
        X = batch[0].to(device)
        y = batch[1].to(device).float().view(-1) # (B,)
        # feat_mask = batch[3].to(device) # Optional: use if model supports it
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(X) # (B,)
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        
        # Accuracy
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
        total_samples += X.size(0)
        
    return total_loss / total_samples, correct / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            X = batch[0].to(device)
            y = batch[1].to(device).float().view(-1)
            
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * X.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total_samples += X.size(0)
            
    return total_loss / total_samples, correct / total_samples

def main():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Loading ---
    # Ensure datasets exist
    if not os.path.exists("data/train_dataset.pth"):
        print("Datasets not found. Generating...")
        prepare_and_save_datasets()

    # Load DataLoaders
    batch_size = 64
    train_loader, val_loader, test_loader, input_dim = get_dataloaders(
        data_dir="data",
        batch_size=batch_size,
        num_workers=0
    )

    print(f"Input Features: {input_dim}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # --- Model Instantiation ---
    # Choose your model: "PTCNet", "ModelA", or "ModelB"
    MODEL_TYPE = "ModelB" 
    print(f"Initializing {MODEL_TYPE}...")

    common_params = dict(
        n_features=input_dim,
        n_blocks=10,
        kernel_size=3,
        dropout=0.1,
        pool_type="none",
        pool_every=1,
        n_domains=None,
        mlp_hidden_dim=256,
        mlp_blocks=4,
    )

    if MODEL_TYPE == "ModelA":
        model = ModelA_NoInteractions(**common_params).to(device)
    elif MODEL_TYPE == "ModelB":
        model = ModelB_WithInteractions(**common_params).to(device)
    else:
        model = PTCNet(**common_params).to(device)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # --- Main Training Loop ---
    num_epochs = 20
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/ptcnet_best.pth"

    print("Starting training...")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("  --> Saved best model")

    # --- Final Test Evaluation ---
    if os.path.exists(checkpoint_path):
        print("Loading best model for testing...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()