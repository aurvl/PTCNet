import matplotlib.pyplot as plt
import numpy as np

def plot_batch_by_label(
    X_batch,
    y_batch,
    max_samples_per_class: int = 3,
    n_features_to_plot: int = 5,
    mode: str = "temporal",
    feature_names: list[str] | None = None,
):
    """
    Visualizes a batch of time-series data, separated by class label.

    Args:
        X_batch: tensor (B, F, T)
        y_batch: tensor (B, 1) or (B,)
        max_samples_per_class: max number of series to plot for y=0 and y=1
        n_features_to_plot: number of features to display per class (for temporal mode)
        mode:
          - "temporal": plot time-series lines
          - "corr": plot correlation heatmaps
        feature_names: optional list of length F for axis labels
    """
    # Ensure CPU and numpy
    X = X_batch.detach().cpu().numpy()
    y = y_batch.detach().cpu().numpy().flatten() # (B,)
    
    # Identify indices for each class
    idxs_0 = np.where(y == 0)[0]
    idxs_1 = np.where(y == 1)[0]
    
    # Select samples
    idxs_0 = idxs_0[:max_samples_per_class]
    idxs_1 = idxs_1[:max_samples_per_class]
    
    # Helper to get effective length (ignore right-padding)
    def get_effective_length(x_sample):
        # x_sample: (F, T)
        # We assume padding is zeros on the right.
        # Find the last column that is NOT all zeros.
        # Sum absolute values across features
        activity = np.sum(np.abs(x_sample), axis=0) # (T,)
        non_zero_indices = np.where(activity > 1e-6)[0]
        if len(non_zero_indices) == 0:
            return 0
        return non_zero_indices[-1] + 1

    if mode == "temporal":
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        
        # --- Class 0 ---
        ax0 = axes[0]
        ax0.set_title(f"Class 0 (Down/Stable) - Showing {len(idxs_0)} samples")
        for idx in idxs_0:
            x_sample = X[idx] # (F, T)
            eff_len = get_effective_length(x_sample)
            # Plot first n features
            for f in range(min(n_features_to_plot, x_sample.shape[0])):
                label = feature_names[f] if feature_names else f"Feat {f}"
                ax0.plot(range(eff_len), x_sample[f, :eff_len], label=label if idx == idxs_0[0] else None, alpha=0.7)
        ax0.grid(True, alpha=0.3)
        if len(idxs_0) > 0:
            ax0.legend(loc="upper right", fontsize="small")

        # --- Class 1 ---
        ax1 = axes[1]
        ax1.set_title(f"Class 1 (Up) - Showing {len(idxs_1)} samples")
        for idx in idxs_1:
            x_sample = X[idx]
            eff_len = get_effective_length(x_sample)
            for f in range(min(n_features_to_plot, x_sample.shape[0])):
                label = feature_names[f] if feature_names else f"Feat {f}"
                ax1.plot(range(eff_len), x_sample[f, :eff_len], label=label if idx == idxs_1[0] else None, alpha=0.7)
        ax1.grid(True, alpha=0.3)
        if len(idxs_1) > 0:
            ax1.legend(loc="upper right", fontsize="small")
            
        plt.tight_layout()
        plt.show()

    elif mode == "corr":
        # Compute correlation matrix for each class
        # We concatenate the time series of all selected samples for that class
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        for i, (class_label, idxs, ax) in enumerate([(0, idxs_0, axes[0]), (1, idxs_1, axes[1])]):
            if len(idxs) == 0:
                ax.text(0.5, 0.5, "No samples", ha='center')
                ax.set_title(f"Class {class_label}")
                continue
                
            # Concatenate effective parts
            data_list = []
            for idx in idxs:
                x_sample = X[idx] # (F, T)
                eff_len = get_effective_length(x_sample)
                if eff_len > 1:
                    data_list.append(x_sample[:, :eff_len])
            
            if not data_list:
                ax.text(0.5, 0.5, "Samples too short", ha='center')
                continue
                
            # Stack along time axis: (F, Total_Time)
            data_concat = np.concatenate(data_list, axis=1)
            
            # Correlation: (F, F)
            # np.corrcoef expects each row to be a variable (feature), so (F, T) is correct
            corr_mat = np.corrcoef(data_concat)
            
            # Plot
            im = ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_title(f"Class {class_label} Correlation ({len(idxs)} samples)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            if feature_names:
                # Show only every nth label if too many
                step = max(1, len(feature_names) // 20)
                ax.set_xticks(np.arange(0, len(feature_names), step))
                ax.set_yticks(np.arange(0, len(feature_names), step))
                ax.set_xticklabels(feature_names[::step], rotation=90, fontsize=8)
                ax.set_yticklabels(feature_names[::step], fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    else:
        print(f"Unknown mode: {mode}")
