# FinancialTSC: Financial Time-Series Classification with PTCNet

This project implements a deep learning pipeline for classifying financial time-series data using **PTCNet** (Pre-Training Context Network), a custom 1D Residual Convolutional Neural Network.

The system is designed to handle **variable-length time series** and **missing features** through dynamic slicing, padding, and masking.

## 📂 Project Structure

```
FinancialTSC/
├── data/                   # Data storage (raw .pkl and processed .pth)
├── notebooks/              # Jupyter notebooks
├── src/
│   ├── feat/               # Data processing & Feature Engineering
│   │   ├── data_collection.py
│   │   ├── data_sequencing.py  # Dataset class & Variable-length Collate fn
│   │   ├── data_lauder.py      # Data loading & Stats
│   │   └── feat_engineer.py
│   ├── ptck_arch/          # Model Architecture
│   │   ├── arch.py             # PTCNet, ModelA, ModelB definitions
│   │   ├── utils.py            # Building blocks (Residual Blocks, MLP)
│   │   └── train_utils.py      # Training step functions
│   └── plot/               # Visualization tools
│       └── data_plot.py
├── notebook.ipynb          # Main experiment notebook (Training + Viz)
├── train.py                # Headless training script
├── requirements.txt        # Dependencies
└── README.md
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd FinancialTSC
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   # Or manually:
   pip install torch torchvision pandas numpy matplotlib scikit-learn
   ```

3. **Install the package in editable mode**:
   This ensures imports like `from src...` work correctly.
   ```bash
   pip install -e .
   ```

## 📊 Data Pipeline

The pipeline is designed to be robust to real-world financial data issues:

1.  **Alignment**: `FinancialDatasetAligned` aligns all assets to a "Master Feature List". Missing features are zero-padded, and a `feature_mask` is generated.
2.  **Dynamic Slicing**: During training, `variable_length_collate` randomly slices the history of each asset (e.g., between 20 and 500 days) to make the model robust to different time horizons.
3.  **Padding**: Short sequences are left-padded with zeros. A `time_mask` is generated to indicate valid time steps.

## 🧠 Model Architectures

The project explores different architectural inductive biases for financial data:

*   **PTCNet (Base)**: A ResNet-like 1D Convolutional backbone followed by an MLP head.
*   **Model A (No Interactions)**: Uses **Depthwise Convolutions** (groups = n_features). Each feature is processed independently in the temporal backbone. Interactions only happen in the final MLP.
*   **Model B (With Interactions)**: Uses standard **Full Convolutions**. Features interact immediately in the temporal backbone.

## 🏃 Usage

### Option 1: Jupyter Notebook (Recommended for Experimentation)
Open `notebook.ipynb` in VS Code or Jupyter Lab. This notebook includes:
*   Data loading and statistics.
*   **Visualization**: Plots time-series samples and correlation matrices for Class 0 vs Class 1.
*   Model instantiation (Choose Model A or B).
*   Training loop with validation and testing.

### Option 2: Training Script (Headless)
For long training runs on a server:
```bash
python train.py
```
This script runs the full training pipeline without visualization code.

### Option 3: Google Colab
If running on Colab:
1.  Upload the `FinancialTSC` folder to Google Drive.
2.  Open `notebook.ipynb` in Colab.
3.  Run the "COLAB / REMOTE SETUP" cell at the top to mount Drive and set the path.

## 📈 Visualization

The `src.plot.data_plot` module provides tools to inspect the data:
*   **Temporal Mode**: Plots the time-series evolution of features for specific classes.
*   **Correlation Mode**: Computes and plots the correlation matrix of features, aggregated over samples of a specific class.

## 🛠️ Configuration

Key hyperparameters can be adjusted in `notebook.ipynb` or `train.py`:
*   `batch_size`: Default 64
*   `seq_len`: Dynamic (20-500)
*   `n_blocks`: Depth of the ResNet backbone (Default 10)
*   `mlp_hidden_dim`: Size of the classification head (Default 256)
