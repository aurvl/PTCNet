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
class DatasetAligned(Dataset):
    def __init__(self, data_dict, master_features):
        """
        Stores the FULL history of each asset.
        Aligns features (Zero-Padding) so every asset has the 'master_features' columns.
        
        Args:
            data_dict (dict): Dictionary of tickers -> {'features': df, 'target': series/df}
            master_features (list): Sorted list of all possible feature names.
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
            
            # C. FEATURE MASK (1 = existait dans ce ticker, 0 = colonne purement padded)
            original_cols = set(df.columns.tolist())
            feat_mask = torch.tensor(
                [1.0 if c in original_cols else 0.0 for c in self.master_features],
                dtype=torch.float32,
            )  # (F,)

            # On stocke maintenant un triplet: (X, y, feat_mask)
            self.samples.append((X_tensor, y_tensor, feat_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Returns the COMPLETE history of the asset at index idx
        # Returns: (X_full, y_full, feat_mask)
        return self.samples[idx]


# --- 3. THE COLLATE FUNCTION (Dynamic Time Slicing) ---
def variable_length_collate(batch):
    """
    Called by the DataLoader for every batch creation.
    Slices time dimension randomly + builds masks.

    Args:
        batch: List of tuples (X_full, y_full, feat_mask)

    Returns:
      X_batch       : (B, F, seq_len)
      y_batch       : (B, 1)
      time_masks    : (B, seq_len)  (1 = vrai timestep, 0 = padding)
      feature_masks : (B, F)
    """
    # A. CHOOSE SEQUENCE LENGTH FOR THIS BATCH
    # We want the model to learn on short windows (20d) and long windows (500d)
    # Pick a random integer between 20 and 500
    seq_len = random.randint(20, 500) 
    
    X_batch = []
    y_batch = []
    time_masks = []
    feat_masks = []
    
    for X_full, y_full, feat_mask in batch:
        # X_full is the complete history (e.g., 4000 days)
        total_time = X_full.shape[1]
        
        # B. SLICING
        if total_time <= seq_len + 1:
            if total_time <= 1:
                continue # Skip degenerate cases
            
            # Rare case: History is too short -> Take everything MINUS ONE
            # We need to reserve the last time step to be the target (y_val)
            start_idx = 0
            actual_len = total_time - 1
        else:
            # Normal case: Slice a chunk of size seq_len randomly
            # Stop before the end to ensure we have a valid target index
            start_idx = random.randint(0, total_time - seq_len - 1)
            actual_len = seq_len
            
        # C. Extract Sequence X and Corresponding Target Y
        X_crop = X_full[:, start_idx : start_idx + actual_len]
        
        # We take the value at the FOLLOWING DAY of the window (decision time)
        target_idx = start_idx + actual_len
        if target_idx >= len(y_full):
            target_idx = len(y_full) - 1  # safeguard
        y_val = y_full[target_idx]
        
        # D. TEMPORAL PADDING (If necessary)
        # If crop is smaller than seq_len (short history case), fill with 0s
        if actual_len < seq_len:
            pad_size = seq_len - actual_len
            # pad = (pad_left, pad_right) sur la dimension temps (la dernière)
            # We pad on the LEFT (past) or RIGHT? 
            # Usually for time series, if we want to align the "latest" point, we pad on the LEFT.
            # But here the user requested: "Pad X_crop on the LEFT with pad_size timesteps"
            # torch.nn.functional.pad(input, (pad_left, pad_right, ...))
            X_crop = torch.nn.functional.pad(
                X_crop, (pad_size, 0), mode="constant", value=0.0
            )
        else:
            pad_size = 0

        # E. TIME MASK : 0 sur le padding, 1 sur les vraies steps
        time_mask = torch.zeros(seq_len, dtype=torch.float32)
        # With left padding, the valid data starts at pad_size
        time_mask[pad_size:] = 1.0

        X_batch.append(X_crop)
        y_batch.append(y_val)
        time_masks.append(time_mask)
        feat_masks.append(feat_mask)
        
    # D. STACKING
    # X Shape: (Batch_Size, N_Features, seq_len) <- seq_len changes every batch!
    # y Shape: (Batch_Size, 1)
    # time_masks: (Batch_Size, seq_len)
    # feature_masks: (Batch_Size, N_Features)
    
    if not X_batch: # Handle empty batch if all samples were skipped
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    return (
        torch.stack(X_batch), 
        torch.stack(y_batch).unsqueeze(1), 
        torch.stack(time_masks), 
        torch.stack(feat_masks)
    )