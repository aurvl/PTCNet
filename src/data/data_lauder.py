import torch
import pickle
import os
from torch.utils.data import DataLoader

from src.feat.data_sequencing import (
    DatasetAligned, variable_length_collate, get_master_feature_list
)

# ==========================================
# 1. BUILDER: Aligns features & Saves to Disk
# ==========================================
def prepare_and_save_datasets(
    train_pkl="data_train.pkl",
    val_pkl="data_val.pkl",
    test_pkl="data_test.pkl",
    output_dir="data"
):
    """
    Loads the raw split dictionaries (pandas), 
    Aligns them to the Master Feature List (Zero-Padding),
    Converts them to PyTorch Tensors,
    and saves the Dataset objects to disk (.pth).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Starting Dataset Preparation...")

    # 1. Load Raw Splits
    print("   Loading pickle files...")
    with open(f"data/{train_pkl}", "rb") as f:
        train_dict = pickle.load(f)
    with open(f"data/{val_pkl}", "rb") as f:
        val_dict = pickle.load(f)
    with open(f"data/{test_pkl}", "rb") as f:
        test_dict = pickle.load(f)

    # 2. Define Master Features (Based on TRAIN only)
    master_features = get_master_feature_list(train_dict)
    print(f"✅ Master Schema: {len(master_features)} features detected.")
    
    # Save Master List (Vital for inference later)
    with open(os.path.join(output_dir, "master_features.pkl"), "wb") as f:
        pickle.dump(master_features, f)

    # 3. Create & Save Datasets
    # This triggers the heavy 'reindex' and 'tensor conversion' logic
    
    print("   Aligning TRAIN set...")
    train_ds = DatasetAligned(train_dict, master_features)
    torch.save(train_ds, os.path.join(output_dir, "train_dataset.pth"))
    
    print("   Aligning VAL set...")
    val_ds = DatasetAligned(val_dict, master_features)
    torch.save(val_ds, os.path.join(output_dir, "val_dataset.pth"))
    
    print("   Aligning TEST set...")
    test_ds = DatasetAligned(test_dict, master_features)
    torch.save(test_ds, os.path.join(output_dir, "test_dataset.pth"))

    print(f"All datasets aligned and saved in '{output_dir}/'. Ready for training!")


# ==========================================
# 2. LOADER: Loads .pth & Returns DataLoaders
# ==========================================
def get_dataloaders(data_dir="data", batch_size=64, num_workers=0):
    """
    To be called in your train.py.
    Loads the pre-aligned datasets and wraps them in DataLoaders 
    with the Dynamic Collate function.
    """
    print(f"Loading Datasets from '{data_dir}'...")
    
    try:
        # Load the objects from disk
        train_ds = torch.load(os.path.join(data_dir, "train_dataset.pth"), weights_only=False)
        val_ds = torch.load(os.path.join(data_dir, "val_dataset.pth"), weights_only=False)
        test_ds = torch.load(os.path.join(data_dir, "test_dataset.pth"), weights_only=False)

        # Load Master Features (to know input_dim)
        with open(os.path.join(data_dir, "master_features.pkl"), "rb") as f:
            master_features = pickle.load(f)
            
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find dataset files. Did you run 'prepare_and_save_datasets'?\n{e}")
        return None, None, None, None

    print(f"   Train size: {len(train_ds)}")
    print(f"   Val size  : {len(val_ds)}")
    print(f"   Test size : {len(test_ds)}")

    # Create DataLoaders
    # This is where the Sequencing happens dynamically (collate_fn)
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=variable_length_collate, # DYNAMIC SLICING HERE
        num_workers=num_workers
    )
    
    # For Validation/Test, we usually want fixed sequences or full history?
    # variable_length_collate to evaluate loss
    # But backtest need a different collate (sequential).
    # For now, we keep the same logic for consistency.
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=variable_length_collate,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=variable_length_collate,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, len(master_features)


def print_dataset_stats(dataset, name="Dataset"):
    """
    Calcule et affiche le nombre total de jours (frames) dans le dataset.
    """
    total_days = 0
    min_days = float('inf')
    max_days = 0
    
    # On parcourt tous les actifs du dataset
    # dataset[i] renvoie (X_tensor, y_tensor)
    # X_tensor shape est (Features, Time)
    for i in range(len(dataset)):
        X, y, _ = dataset[i]
        time_steps = X.shape[1] # La dim temporelle
        
        total_days += time_steps
        if time_steps < min_days: 
            min_days = time_steps
        if time_steps > max_days: 
            max_days = time_steps
        
    print(f"📊 Statistiques pour {name}:")
    print(f"   ├── Nombre d'actifs : {len(dataset)}")
    print(f"   └── Total Jours (Frames) : {total_days:,} jours")
    print(f"    ├── Plus petit historique : {min_days} jours")
    print(f"    └── Plus grand historique : {max_days} jours")
    print("-" * 30)

def print_target_distribution(dataset, name="Dataset"):
    """
    Calculates and prints the distribution of the binary target variable (0 vs 1).
    Handles NaN values gracefully.
    """
    count_0 = 0
    count_1 = 0
    count_nan = 0
    total_samples = 0
    
    # Iterate through all assets in the dataset
    for i in range(len(dataset)):
        _, y, _ = dataset[i]
        # y is a tensor of shape (Time,) containing 0s and 1s
        
        # Handle NaNs
        if torch.isnan(y).any():
            nan_mask = torch.isnan(y)
            n_nans = nan_mask.sum().item()
            count_nan += n_nans
            
            # Filter valid
            y_valid = y[~nan_mask]
            c1 = y_valid.sum().item()
            c0 = y_valid.numel() - c1
        else:
            c1 = y.sum().item()
            c0 = y.numel() - c1
        
        count_1 += c1
        count_0 += c0
        total_samples += y.numel()
        
    valid_samples = count_0 + count_1
    
    if valid_samples == 0:
        print(f"⚠️ No valid samples found in {name}.")
        if total_samples > 0:
             print(f"   (Found {int(total_samples)} NaNs)")
        return

    pct_0 = (count_0 / valid_samples) * 100
    pct_1 = (count_1 / valid_samples) * 100
    
    print(f"⚖️ Target Distribution for {name}:")
    print(f"   ├── Total Targets : {int(total_samples):,}")
    if count_nan > 0:
        print(f"   ├── NaNs Found    : {int(count_nan):,} ({(count_nan/total_samples)*100:.2f}%)")
    print(f"   ├── Class 0 (Down/Stable) : {int(count_0):,} ({pct_0:.2f}%)")
    print(f"   └── Class 1 (Up)          : {int(count_1):,} ({pct_1:.2f}%)")
    print("-" * 30)


# --- Execution Check ---
if __name__ == "__main__":
    # 1. Run Prep (Only once)
    # prepare_and_save_datasets()
    
    # 2. Test Loading
    t_loader, v_loader, test_loader, input_dim = get_dataloaders(batch_size=16)
    
    train_ds = t_loader.dataset
    val_ds = v_loader.dataset
    test_ds = test_loader.dataset

    print("\nChecking Batch generation...")
    X, y, time_masks, feat_masks = next(iter(t_loader))
    print(f"Input Dim: {input_dim}")
    print(f"Batch X shape: {X.shape} (Time is dynamic!)")
    print(f"Batch y shape: {y.shape}")
    print(f"Batch time_masks shape: {time_masks.shape}")
    print(f"Batch feat_masks shape: {feat_masks.shape}")
    
    print("\nANALYSE DU VOLUME DE DONNÉES :")
    print_dataset_stats(train_ds, "TRAIN")
    print_dataset_stats(val_ds, "VAL")
    print_dataset_stats(test_ds, "TEST")

    print("\nANALYSE DE LA DISTRIBUTION DES CIBLES :")
    print_target_distribution(train_ds, "TRAIN")
    print_target_distribution(val_ds, "VAL")
    print_target_distribution(test_ds, "TEST")