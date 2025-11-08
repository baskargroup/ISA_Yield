import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error  # Added MAE
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import pdb
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
    X_list, y_list, file_label_list = [], [], []
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
        mask = np.isfinite(y)
        X = X[mask]
        y = y[mask]
        file_labels = np.array([fname] * len(y))
        X_list.append(X)
        y_list.append(y)
        file_label_list.append(file_labels)
    if X_list:
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        file_labels = np.concatenate(file_label_list, axis=0)
    else:
        X = np.empty((0, 1))
        y = np.empty((0,))
        file_labels = np.empty((0,), dtype=object)
    nonzero_mask = y != 0
    X = X[nonzero_mask]
    y = y[nonzero_mask]
    file_labels = file_labels[nonzero_mask]
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

    # # --- Random Forest ---
    # rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    # rf.fit(X_train, y_train)
    # y_pred_train_rf = rf.predict(X_train)
    # y_pred_val_rf = rf.predict(X_val)
    # y_pred_test_rf = rf.predict(X_test)
    # results.append({
    #     "biweek": biweek_folder,
    #     "model": "RandomForest",
    #     "train_r2": r2_score(y_train, y_pred_train_rf),
    #     "val_r2": r2_score(y_val, y_pred_val_rf),
    #     "test_r2": r2_score(y_test, y_pred_test_rf),
    #     "train_rmse": root_mean_squared_error(y_train, y_pred_train_rf),
    #     "val_rmse": root_mean_squared_error(y_val, y_pred_val_rf),
    #     "test_rmse": root_mean_squared_error(y_test, y_pred_test_rf)
    # })

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
    "train_mae": mean_absolute_error(y_train, y_pred_train_xgb),   # Added MAE
    "test_mae": mean_absolute_error(y_test, y_pred_test_xgb)       # Added MAE
})


    # Aggregate metrics by file
    df_xgb = pd.DataFrame({
        "file": test_labels,
        "y_true": y_test,
        "y_pred": y_pred_test_xgb
    })
    file_level_xgb = []
    for fname, group in df_xgb.groupby("file"):
        file_level_xgb.append({
            "biweek": biweek_folder,
            "model": "XGBoost",
            "file": fname,
            "r2": r2_score(group["y_true"], group["y_pred"]),
            "rmse": root_mean_squared_error(group["y_true"], group["y_pred"]),
            "mae": mean_absolute_error(group["y_true"], group["y_pred"])
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
        "train_mae": mean_absolute_error(y_train, y_pred_train_pls),   # Added MAE
        "test_mae": mean_absolute_error(y_test, y_pred_test_pls)       # Added MAE
    })

    df_pls = pd.DataFrame({
        "file": test_labels,
        "y_true": y_test,
        "y_pred": y_pred_test_pls
    })
    file_level_pls = []
    for fname, group in df_pls.groupby("file"):
        file_level_pls.append({
            "biweek": biweek_folder,
            "model": "PLSR",
            "file": fname,
            "r2": r2_score(group["y_true"], group["y_pred"]),
            "rmse": root_mean_squared_error(group["y_true"], group["y_pred"]),
            "mae": mean_absolute_error(group["y_true"], group["y_pred"])
        })

    # Collect file-level results
    if biweek_folder == all_biweek_folders[0]:
        file_level_results = []
    file_level_results.extend(file_level_xgb)
    file_level_results.extend(file_level_pls)

# Save results as a table
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv(f"classical_ml_results_{_code}.csv", index=False)

# Save file-level results
df_file_level = pd.DataFrame(file_level_results)
print(df_file_level)
df_file_level.to_csv(f"classical_ml_file_level_results_{_code}.csv", index=False)