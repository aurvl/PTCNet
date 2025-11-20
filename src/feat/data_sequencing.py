import torch
from torch.utils.data import Dataset
import random

# --- 1. UTILITY FUNCTION ---
def get_master_feature_list(data_dict):
    """
    Scans the entire dictionary (Train set) to find the union of all possible columns.
    This defines the input dimension (input_channels) of your model.
    """
    all_cols = set()
    for content in data_dict.values():
        # Retrieve columns for each asset
        all_cols.update(content['features'].columns.tolist())
    
    # Sort to ensure the order is always identical (Vital for the CNN)
    return sorted(list(all_cols))


# --- 2. THE DATASET (Spatial Alignment / Zero-Padding) ---
class FinancialDatasetAligned(Dataset):
    def __init__(self, data_dict, master_features):
        """
        Stores the FULL history of each asset.
        Aligns features (Zero-Padding) so every asset has the 'master_features' columns.
        """
        self.samples = []
        self.master_features = master_features
        self.feature_count = len(master_features)
        
        print(f"⚙️ Aligning data to {self.feature_count} features...")
        
        for ticker, content in data_dict.items():
            df = content['features']
            target = content['target']
            
            # A. ALIGNMENT (Zero Padding)
            # Fills the missing features with 0.0
            # If the asset doesn't have the 'GDP' feature, it gets 0.0 everywhere.
            df_aligned = df.reindex(columns=self.master_features, fill_value=0.0)
            
            # B. CONVERSION TO TENSOR (FULL HISTORY)
            # Conv1d expects (Batch, Channels, Time)
            # So we transpose: (Time, Feat) -> (Feat, Time)
            X_tensor = torch.tensor(df_aligned.values, dtype=torch.float32).T
            
            # Target
            y_tensor = torch.tensor(target.values, dtype=torch.float32)
            
            self.samples.append((X_tensor, y_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Returns the COMPLETE history of the asset at index idx
        return self.samples[idx]


# --- 3. THE COLLATE FUNCTION (Dynamic Time Slicing) ---
def variable_length_collate(batch):
    """
    Called by the DataLoader for every batch creation.
    This is where we slice the time dimension RANDOMLY.
    """
    # A. CHOOSE SEQUENCE LENGTH FOR THIS BATCH
    # We want the model to learn on short windows (20d) and long windows (150d)
    # Pick a random integer between 20 and 150
    seq_len = random.randint(20, 150) 
    
    X_batch = []
    y_batch = []
    
    for X_full, y_full in batch:
        # X_full is the complete history (e.g., 4000 days)
        total_time = X_full.shape[1]
        
        # B. SLICING
        if total_time <= seq_len:
            # Rare case: History is too short -> Take everything
            start_idx = 0
            actual_len = total_time
        else:
            # Normal case: Slice a chunk of size seq_len randomly
            # Stop before the end to ensure we have a valid target index
            start_idx = random.randint(0, total_time - seq_len - 1)
            actual_len = seq_len
            
        # Extract Sequence X
        X_crop = X_full[:, start_idx : start_idx + actual_len]
        
        # Extract Corresponding Target Y
        # We take the value at the FOLLOWING DAY of the window (decision time)
        y_val = y_full[start_idx + actual_len]
        
        # C. TEMPORAL PADDING (If necessary)
        # If crop is smaller than seq_len (short history case), fill with 0s
        if X_crop.shape[1] < seq_len:
             pad_size = seq_len - X_crop.shape[1]
             # Pad takes (padding_left, padding_right, padding_top, padding_bottom...)
             # We pad the Time dimension (the last one) on the right
             X_crop = torch.nn.functional.pad(X_crop, (0, pad_size), "constant", 0)

        X_batch.append(X_crop)
        y_batch.append(y_val)
        
    # D. STACKING
    # X Shape: (Batch_Size, N_Features, seq_len) <- seq_len changes every batch!
    # y Shape: (Batch_Size, 1)
    return torch.stack(X_batch), torch.stack(y_batch).unsqueeze(1)