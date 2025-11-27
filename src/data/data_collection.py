import random
import time
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.feat.feat_engineer import (  # noqa: E402
    get_data_yf, f1, f2, f3, f4, f5, f6, create_y, 
    compute_dynamic_k_series, load_context,
    load_gdp_context
)

def generate_pretraining_dataset(save_path="processed_data_diverse.pkl"):
    """
    Generates a diverse pre-training dataset (Stocks, Forex, Indices, Commodities).
    
    Process:
    1. Iterates through a diverse list of assets.
    2. Randomly applies a feature engineering strategy (f1...f6).
    3. Randomly selects a prediction horizon (21 to 126 days).
    4. Generates the target variable (Y) based on dynamic volatility thresholds.
    5. Saves the result as a pickle file.
    """
    
    # 1. Diverse Asset List (Yahoo Finance Tickers)
    tickers = [
        # --- 1. US MEGA CAPS (TECH & GROWTH) ---
        # Learns strong trends and bubbles
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", 
        "AMD", "INTC", "QCOM", "AVGO", "TXN", "CRM", "ADBE", "ORCL", 
        "CSCO", "IBM", "UBER", "ABNB", "PLTR", "SNOW", "SHOP",

        # --- 2. US BLUE CHIPS (VALUE & DEFENSIVE) ---
        # Learns stability and economic cycles
        "BRK-B", "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", # Finance
        "PG", "KO", "PEP", "MCD", "WMT", "COST", "TGT", "HD", "LOW",     # Consumption
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "AMGN", "CVS",        # Healthcare
        "XOM", "CVX", "COP", "SLB", "EOG",                               # Energy
        "BA", "CAT", "DE", "GE", "HON", "MMM", "UNP", "UPS",             # Industry
        "DIS", "CMCSA", "VZ", "T",                                       # Telecom/Media

        # --- 3. GLOBAL & EUROPE (ADRs & Direct) ---
        # Geographic diversification
        "TSM", "ASML", "SAP", "SONY", "TM", "HMC", "BABA", "JD", "BIDU", # Asia/Tech
        "NVO", "AZN", "NVS",                                             # Pharma Euro
        "SHEL", "BP", "TTE",                                             # Energy Euro
        "HSBC", "RY", "TD",                                              # Global Banks
        "UL", "BUD", "DEO",                                              # Global Consumption

        # --- 4. SECTOR ETFs (SPDR) ---
        # Represents entire sectors (less noise than individual stocks)
        "XLE", # Energy
        "XLF", # Finance
        "XLK", # Technology
        "XLV", # Healthcare
        "XLI", # Industrial
        "XLP", # Consumer Staples
        "XLY", # Consumer Discretionary
        "XLU", # Utilities (Very stable)
        "XLB", # Materials
        "XLRE", # Real Estate

        # --- 5. GLOBAL INDICES & VOLATILITY ---
        "SPY", "QQQ", "IWM", "DIA", # US Indices
        "EEM", "EFA", "FXI", "EWZ", # Emerging Markets, Europe/Asia, China, Brazil
        "VGK", "EWJ",               # Europe, Japan
        "^VIX",                     # The Fear Index (Crucial!)

        # --- 6. COMMODITIES ---
        # Learns inflation and macro cycles
        "GC=F", # Gold
        "SI=F", # Silver
        "CL=F", # Crude Oil WTI
        "BZ=F", # Crude Oil Brent
        "NG=F", # Natural Gas
        "HG=F", # Copper
        "PL=F", # Platinum
        "ZC=F", # Corn
        "ZW=F", # Wheat
        "ZS=F", # Soybean

        # --- 7. FOREX (Currencies) ---
        # Learns "Mean Reversion" (Oscillates around a value)
        "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
        "USDCAD=X", "USDCHF=X", "EURGBP=X", "EURJPY=X",

        # --- 8. BONDS & RATES (The "Risk Free Rate") ---
        # Learns the cost of money
        "^TNX", # 10 Year Treasury Yield
        "^TYX", # 30 Year Treasury Yield
        "^IRX", # 13 Week Treasury Bill
        "TLT",  # 20+ Year Treasury Bond ETF
        "IEF",  # 7-10 Year Treasury Bond ETF
        "SHY",  # 1-3 Year Treasury Bond ETF
        "LQD",  # Investment Grade Corporate Bonds
        "HYG",  # High Yield Corporate Bonds (Junk Bonds - Risk On)

        # --- 9. CRYPTO (Optional but good for extreme volatility) ---
        # Learns to handle -50% crashes and bubbles
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"
    ]

    # List of feature engineering functions
    functions_list = [f1, f2, f3, f4, f5, f6]
    
    processed_data = {}
    
    print(f"Starting processing for {len(tickers)} assets...")
    print("Loading Macro Contexts (GDP, VIX)...")
    
    # OPTIMIZATION: Load context data ONLY ONCE before the loop
    try:
        # Ensure paths are correct relative to where you run the script
        df_bench, df_vix = load_context('data/context.csv')
        gdps = load_gdp_context('data/GDP.csv', 1990)
    except Exception as e:
        print(f"Critical Error loading global context: {e}")
        return {}

    print("-" * 65)
    print(f"{'TICKER':<10} | {'ROWS':>6} | {'FUNC':<4} | {'HZ':>4} | {'TIME':>6}")
    print("-" * 65)

    for ticker in tickers:
        try:
            start_time = time.time()
            
            # 1. Download Data
            # We take a wide range to ensure enough history for long windows
            df = get_data_yf(ticker, start="2000-01-01")
            
            if df.empty or len(df) < 300: # Skip if not enough data for training
                print(f"Not enough data for {ticker}")
                continue
            
            # 2. Random Strategy Selection
            selected_func = random.choice(functions_list)
            func_name = selected_func.__name__

            # 3. Apply Feature Engineering (Switch Case logic)
            # Passing the pre-loaded context variables
            if func_name == 'f1':
                result_df = selected_func(df, df_bench=df_bench, df_vix=df_vix)
            elif func_name == 'f2':
                result_df = selected_func(df, df_vix=df_vix)
            elif func_name == 'f5':
                # Note: f5 requires specific argument name 'df_gdp'
                result_df = selected_func(df, df_vix=df_vix, df_gdp=gdps)
            elif func_name == 'f4':
                 result_df = selected_func(df, df_vix=df_vix)
            else:
                # f3, f6 and generic functions
                result_df = selected_func(df)
            
            # 4. Dynamic Target Creation
            # Randomly select a horizon for THIS specific stock instance
            random_horizon = random.randint(21, 126) # Between 1 month and 6 months
            
            # Lookback for local volatility calculation (needs to be larger than horizon)
            lookback = max(60, 3 * random_horizon) 
            
            # Compute dynamic return threshold (k) based on volatility
            k_series = compute_dynamic_k_series(df, T_H=random_horizon, window=lookback)
            
            # Create Target Label Y
            # IMPORTANT: Pass T_H=random_horizon so create_y looks at the correct future date
            result_target = create_y(df_price=df, k=k_series, T_H=random_horizon)
            
            # 5. Store Result
            processed_data[ticker] = {
                "type": "Asset", 
                "function_used": func_name,
                "features": result_df,
                "target": result_target,
                "horizon": random_horizon
            }
            
            elapsed_time = time.time() - start_time
            print(f"✅ {ticker:<10} | {len(df):>6} | {func_name:<4} | {random_horizon:>3}d | {elapsed_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error on {ticker} : {e}")
            continue

    print("-" * 65)
    print(f"Processing complete. {len(processed_data)} assets processed successfully.")
    
    # 6. Save to Pickle
    if save_path:
        print(f"Saving data to {save_path}...")
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(processed_data, f)
            print("Save successful!")
        except Exception as e:
            print(f"❌ Error saving pickle file: {e}")

    return processed_data

def process_split_and_scale(
    data_dict: dict, 
    train_end: str = "2017-01-01", 
    val_end: str = "2020-01-01"
):
    """
    Splits data into Train/Val/Test and applies StandardScaler.
    
    CRITICAL: The Scaler is FITTED on TRAIN data only, then APPLIED to Val and Test.
    This prevents data leakage (looking into the future).
    
    Args:
        data_dict: The dictionary containing features and targets.
        train_end: Cutoff date for training (exclusive).
        val_end: Cutoff date for validation (exclusive).
        
    Returns:
        train_dict, val_dict, test_dict, scalers_dict
    """
    print("Splitting & Scaling Data...")
    print(f"   Train: < {train_end}")
    print(f"   Val  : {train_end} to {val_end}")
    print(f"   Test : >= {val_end}")
    
    train_data = {}
    val_data = {}
    test_data = {}
    scalers = {} # We keep scalers if we need to unscale later
    
    skipped_count = 0
    
    for ticker, content in data_dict.items():
        # 1. Extract Data
        df = content['features']
        target = content['target']
        target = target.reindex(df.index)
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            target.index = pd.to_datetime(target.index)
            
        # 2. Create Masks
        mask_train = (df.index < train_end)
        mask_val   = (df.index >= train_end) & (df.index < val_end)
        mask_test  = (df.index >= val_end)
        
        # 3. Slice DataFrames
        X_train = df[mask_train]
        y_train = target[mask_train]
        
        X_val   = df[mask_val]
        y_val   = target[mask_val]
        
        X_test  = df[mask_test]
        y_test  = target[mask_test]

        # --- FIX: DROP NaNs in Targets ---
        # Targets can be NaN at the end of the series (horizon lookahead).
        # We must remove these rows from both X and y.
        
        # Train
        valid_train = ~y_train.isna()
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        # Val
        valid_val = ~y_val.isna()
        X_val = X_val[valid_val]
        y_val = y_val[valid_val]

        # Test
        valid_test = ~y_test.isna()
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        # ---------------------------------
        
        # 4. Safety Check
        # If a stock is too recent (IPO in 2019), it has no Train data.
        # We CANNOT train a scaler on it. We must skip it for training.
        if len(X_train) < 50:
            # Optional: If it has Val/Test data, we could use it for testing only 
            # BUT we wouldn't have a scaler fitted. 
            # Simple strategy: Skip stocks that don't exist in the training era.
            skipped_count += 1
            continue
            
        # 5. SCALING (The Important Part)
        scaler = StandardScaler()
        
        # FIT only on TRAIN
        scaler.fit(X_train)
        
        # TRANSFORM all sets
        # scaler.transform returns numpy array, we must rebuild DataFrame to keep columns/index
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
        
        # Check if Val/Test are not empty before transforming
        if not X_val.empty:
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
        else:
            X_val_scaled = pd.DataFrame(columns=X_train.columns)

        if not X_test.empty:
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_train.columns)
        else:
            X_test_scaled = pd.DataFrame(columns=X_train.columns)
            
        # 6. Store in Dictionaries
        # We keep the metadata (function used, horizon) for the Dataset loader
        meta = {
            'function_used': content['function_used'],
            'horizon': content['horizon']
        }
        
        train_data[ticker] = {**meta, 'features': X_train_scaled, 'target': y_train}
        
        if not X_val_scaled.empty:
            val_data[ticker] = {**meta, 'features': X_val_scaled, 'target': y_val}
            
        if not X_test_scaled.empty:
            test_data[ticker] = {**meta, 'features': X_test_scaled, 'target': y_test}
            
        # Save the scaler (useful for inference later)
        scalers[ticker] = scaler

    print("Processing Done!")
    print(f"   Train Stocks: {len(train_data)}")
    print(f"   Val Stocks  : {len(val_data)}")
    print(f"   Test Stocks : {len(test_data)}")
    print(f"   Skipped (No Train Data): {skipped_count}")
    
    return train_data, val_data, test_data, scalers


# --- ENTRY POINT ---
if __name__ == "__main__":
    # Run the generator
    # data = generate_pretraining_dataset(save_path="data/processed_data_diverse.pkl")
    
    # # Optional: Print a sample to verify
    # if data:
    #     first_key = list(data.keys())[0]
    #     print(f"\nSample Data ({first_key}):")
    #     print(f"Strategy: {data[first_key]['function_used']}")
    #     print(f"Horizon: {data[first_key]['horizon']} days")
    #     print(f"Features Shape: {data[first_key]['features'].shape}")

    print("\n\nLoading pickle...")
    with open("data/processed_data_diverse.pkl", "rb") as f:
        raw_data = pickle.load(f)
        
    # Run the split & scale
    train, val, test, scalers = process_split_and_scale(raw_data)
    # train, val, test, scalers = process_split_and_scale(data)
    
    # Save the splits (Ready for DataLoader)
    with open("data/data_train.pkl", "wb") as f: 
        pickle.dump(train, f)
    with open("data/data_val.pkl", "wb") as f: 
        pickle.dump(val, f)
    with open("data/data_test.pkl", "wb") as f: 
        pickle.dump(test, f)
    with open("data/scalers.pkl", "wb") as f: 
        pickle.dump(scalers, f)
    
    print("All datasets saved!")