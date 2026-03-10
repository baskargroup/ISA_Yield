import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
try:
    import cupy as cp
    use_cupy = True
except ImportError:
    use_cupy = False
# List all processed_data_weekly_* folders
DATA_ROOT = 'processed_data/weekly_24'
all_biweek_folders = sorted([os.path.join(DATA_ROOT, f) for f in os.listdir(DATA_ROOT)
                            if f.startswith('processed_data_weekly') and
                            os.path.isdir(os.path.join(DATA_ROOT, f)) and
                            not f.endswith('_selected') and not f.endswith('_selected_8')])
modalities = ["S2L2A", 
              "S1GRD",
              "CDL",
              "DEM",
              "WEATHER",
              ]

_code = "s12cdw"
label_folder = "yield_geotiffs"

results = []
file_level_results = []

def read_split(split_path):
    with open(split_path) as f:
        return [line.strip() + ".npy" for line in f if line.strip()]

def build_X_y(file_names, modal_paths, label_path):
    aggregated_rows = []
    
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
        # Filter finite values and non-zero yields
        mask = np.isfinite(y) & (y != 0)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Skip if no valid pixels
        if len(y_filtered) == 0:
            continue
        
        # Aggregate this file immediately - compute mean for all features and yield
        # Use nanmean to ignore NaN values when computing the mean
        X_mean = np.nanmean(X_filtered, axis=0)
        y_mean = np.nanmean(y_filtered)
        
        # Skip if all values are NaN (results in NaN after nanmean)
        if np.any(np.isnan(X_mean)) or np.isnan(y_mean):
            continue
        
        # Create aggregated row for this file
        row = {f'feature_{j}': X_mean[j] for j in range(len(X_mean))}
        row['yield'] = y_mean
        row['file'] = basename
        aggregated_rows.append(row)
    
    # Create dataframe from aggregated rows
    df_agg = pd.DataFrame(aggregated_rows)
    
    if len(df_agg) == 0:
        # Return empty arrays if no data
        return np.empty((0, 1)), np.empty((0,)), np.empty((0,), dtype=object)
    
    # Extract X, y, and file_labels from aggregated dataframe
    feature_cols = [col for col in df_agg.columns if col.startswith('feature_')]
    X = df_agg[feature_cols].values
    y = df_agg['yield'].values
    file_labels = df_agg['file'].values
    return X, y, file_labels

for biweek_folder in all_biweek_folders:
    week_name = os.path.basename(biweek_folder)
    print(f"\n=== Processing {week_name} ===")
    # Prepare paths
    splits = {}
    for split in ["train", "val", "test"]:
        split_file = os.path.join(biweek_folder, f"{split}_corn.txt")
        if not os.path.exists(split_file):
            split_file = os.path.join(biweek_folder, split, f"{split}_corn.txt")
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
    xgb.fit(X_train, y_train)
    y_pred_train_xgb = xgb.predict(X_train)
    y_pred_test_xgb = xgb.predict(X_test)
    
    results.append({
        "week": week_name,
        "model": "XGBoost",
        "train_r2": r2_score(y_train, y_pred_train_xgb),
        "test_r2": r2_score(y_test, y_pred_test_xgb),
        "train_rmse": root_mean_squared_error(y_train, y_pred_train_xgb),
        "test_rmse": root_mean_squared_error(y_test, y_pred_test_xgb),
        "train_mae": mean_absolute_error(y_train, y_pred_train_xgb),
        "test_mae": mean_absolute_error(y_test, y_pred_test_xgb),
        "train_mape": mean_absolute_percentage_error(y_train, y_pred_train_xgb),
        "test_mape": mean_absolute_percentage_error(y_test, y_pred_test_xgb)
    })

    # Save field-level XGBoost predictions
    for fname, yt, yp in zip(test_labels, y_test, y_pred_test_xgb):
        file_level_results.append({
            "week": week_name, "model": "XGBoost", "file": fname,
            "y_true": yt, "y_pred": yp,
        })

    # --- PLSR ---
    n_components = min(20, X_train.shape[1])
    plsr = PLSRegression(n_components=n_components)
    plsr.fit(X_train, y_train)
    y_pred_train_pls = plsr.predict(X_train).ravel()
    y_pred_test_pls = plsr.predict(X_test).ravel()
    results.append({
        "week": week_name,
        "model": "PLSR",
        "train_r2": r2_score(y_train, y_pred_train_pls),
        "test_r2": r2_score(y_test, y_pred_test_pls),
        "train_rmse": root_mean_squared_error(y_train, y_pred_train_pls),
        "test_rmse": root_mean_squared_error(y_test, y_pred_test_pls),
        "train_mae": mean_absolute_error(y_train, y_pred_train_pls),
        "test_mae": mean_absolute_error(y_test, y_pred_test_pls),
        "train_mape": mean_absolute_percentage_error(y_train, y_pred_train_pls),
        "test_mape": mean_absolute_percentage_error(y_test, y_pred_test_pls)
    })

    # Save field-level PLSR predictions
    for fname, yt, yp in zip(test_labels, y_test, y_pred_test_pls):
        file_level_results.append({
            "week": week_name, "model": "PLSR", "file": fname,
            "y_true": yt, "y_pred": yp,
        })

# Save results
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv(f"classical_ml_results_corn_{_code}.csv", index=False)

# Save field-level predictions
df_file = pd.DataFrame(file_level_results)
df_file.to_csv(f"classical_ml_field_level_corn_{_code}.csv", index=False)
print(f"\nSaved {len(df_results)} aggregate rows, {len(df_file)} field-level rows")