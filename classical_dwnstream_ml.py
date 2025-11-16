import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
try:
    import cupy as cp
    use_cupy = True
except ImportError:
    use_cupy = False
# List all processed_data_biweekly_* folders
all_biweek_folders = sorted([f for f in os.listdir('.') if f.startswith('processed_data_biweekly') and os.path.isdir(f)])
modalities = ["S2L2A", 
              "S1GRD",
              "DEM", 
              "WEATHER",
              ]

_code = "s12wd"
label_folder = "yield_geotiffs"

results = []

def read_split(split_path):
    with open(split_path) as f:
        return [line.strip() + ".npy" for line in f if line.strip()]

def build_X_y(file_names, modal_paths, label_path):
    data_rows = []
    
    for fname in tqdm(file_names, desc="Files"):
        feats = []
        skip = False
        for m in modalities:
            mfile = os.path.join(modal_paths[m], fname)
            if not os.path.exists(mfile):
                skip = True
                break
            arr = np.load(mfile)
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
            arr = arr.transpose(2, 0, 1).reshape(-1, arr.shape[0]*arr.shape[1])
            feats.append(arr)
        if skip:
            continue
        X = np.concatenate(feats, axis=1)
        y = np.load(os.path.join(label_path, fname))
        if y.ndim == 3:
            y = y.squeeze(0)
        y = y.flatten()
        
        # Get basename without extension
        basename = os.path.splitext(fname)[0]
        # Filter finite values
        mask = np.isfinite(y)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Create rows for dataframe
        for i in range(len(y_filtered)):
            row = {f'feature_{j}': X_filtered[i, j] for j in range(X_filtered.shape[1])}
            row['yield'] = y_filtered[i]
            row['file'] = basename
            data_rows.append(row)
    
    # Create dataframe
    df = pd.DataFrame(data_rows)
    
    if len(df) == 0:
        # Return empty arrays if no data
        return np.empty((0, 1)), np.empty((0,)), np.empty((0,), dtype=object)
    
    # Discard rows where yield is zero
    df = df[df['yield'] != 0]
    
    # Aggregate by file - compute mean for all features and yield
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    agg_dict = {col: 'mean' for col in feature_cols}
    agg_dict['yield'] = 'mean'
    
    df_agg = df.groupby('file', as_index=False).agg(agg_dict)
    # Extract X, y, and file_labels from aggregated dataframe
    X = df_agg[feature_cols].values
    y = df_agg['yield'].values
    file_labels = df_agg['file'].values
    return X, y, file_labels

for biweek_folder in all_biweek_folders:
    print(f"\n=== Processing {biweek_folder} ===")
    # Prepare paths
    splits = {}
    for split in ["train", "val", "test"]:
        split_file = os.path.join(biweek_folder, f"{split}.txt")
        if not os.path.exists(split_file):
            split_file = os.path.join(biweek_folder, split, f"{split}.txt")
        splits[split] = read_split(split_file)
    modal_paths = {m: os.path.join(biweek_folder, m) for m in modalities}
    label_path = os.path.join(biweek_folder, label_folder)

    # Build data
    X_train, y_train, train_labels = build_X_y(splits["train"], modal_paths, label_path)
    X_test, y_test, test_labels = build_X_y(splits["test"], modal_paths, label_path)

    # --- XGBoost ---
    xgb_params = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": 42,
        "tree_method": "hist",
        # "device": "cuda",
    }
    xgb = XGBRegressor(**xgb_params)
    if use_cupy:
        X_train_xgb = cp.asarray(X_train)
        X_test_xgb = cp.asarray(X_test)
        y_train_xgb = cp.asarray(y_train)
        y_test_xgb = cp.asarray(y_test)
    else:
        X_train_xgb = X_train
        X_test_xgb = X_test
        y_train_xgb = y_train
        y_test_xgb = y_test

    xgb.fit(X_train_xgb, y_train_xgb)
    y_pred_train_xgb = xgb.predict(X_train_xgb)
    y_pred_test_xgb = xgb.predict(X_test_xgb)
    if use_cupy:
        y_pred_test_xgb = cp.asnumpy(y_pred_test_xgb)
    
    results.append({
        "biweek": biweek_folder,
        "model": "XGBoost",
        "train_r2": r2_score(y_train, y_pred_train_xgb),
        "test_r2": r2_score(y_test, y_pred_test_xgb),
        "train_rmse": root_mean_squared_error(y_train, y_pred_train_xgb),
        "test_rmse": root_mean_squared_error(y_test, y_pred_test_xgb),
        "train_mae": mean_absolute_error(y_train, y_pred_train_xgb),
        "test_mae": mean_absolute_error(y_test, y_pred_test_xgb)
    })

    # --- PLSR ---
    n_components = min(20, X_train.shape[1])
    plsr = PLSRegression(n_components=n_components)
    plsr.fit(X_train, y_train)
    y_pred_train_pls = plsr.predict(X_train).ravel()
    y_pred_test_pls = plsr.predict(X_test).ravel()
    results.append({
        "biweek": biweek_folder,
        "model": "PLSR",
        "train_r2": r2_score(y_train, y_pred_train_pls),
        "test_r2": r2_score(y_test, y_pred_test_pls),
        "train_rmse": root_mean_squared_error(y_train, y_pred_train_pls),
        "test_rmse": root_mean_squared_error(y_test, y_pred_test_pls),
        "train_mae": mean_absolute_error(y_train, y_pred_train_pls),
        "test_mae": mean_absolute_error(y_test, y_pred_test_pls)
    })

# Save results as a table
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv(f"classical_ml_results_{_code}.csv", index=False)