import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
try:
    import cupy as cp
    use_cupy = True
except ImportError:
    use_cupy = False

# Configuration
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

# Denormalization: norm = (raw - data_min) / (data_max - data_min)
DATA_MIN, DATA_MAX = 50.0, 370.0

def denormalize(y_norm):
    return y_norm * (DATA_MAX - DATA_MIN) + DATA_MIN

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
        
        basename = os.path.splitext(fname)[0]
        mask = np.isfinite(y) & (y != 0)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        if len(y_filtered) == 0:
            continue
        
        X_mean = np.nanmean(X_filtered, axis=0)
        y_mean = np.nanmean(y_filtered)
        
        if np.any(np.isnan(X_mean)) or np.isnan(y_mean):
            continue
        
        row = {f'feature_{j}': X_mean[j] for j in range(len(X_mean))}
        row['yield'] = y_mean
        row['file'] = basename
        aggregated_rows.append(row)
    
    df_agg = pd.DataFrame(aggregated_rows)
    
    if len(df_agg) == 0:
        return np.empty((0, 1)), np.empty((0,)), np.empty((0,), dtype=object)
    
    feature_cols = [col for col in df_agg.columns if col.startswith('feature_')]
    X = df_agg[feature_cols].values
    y = df_agg['yield'].values
    file_labels = df_agg['file'].values
    return X, y, file_labels

# Define scoring metric for cross-validation
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

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
    print("Building training data...")
    X_train, y_train, train_labels = build_X_y(splits["train"], modal_paths, label_path)
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")

    print("Building test data...")
    X_test, y_test, test_labels = build_X_y(splits["test"], modal_paths, label_path)
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping {week_name}: insufficient data")
        continue

    # ============ XGBoost Parameter Search ============
    print("\n" + "="*50)
    print("Searching for best XGBoost parameters...")
    print("="*50)

    xgb_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }

    xgb_base = XGBRegressor(
        n_jobs=-1,
        random_state=42,
        tree_method="hist"
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_grid,
        n_iter=50,
        scoring=rmse_scorer,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    xgb_search.fit(X_train, y_train)

    print(f"\nBest XGBoost parameters: {xgb_search.best_params_}")
    print(f"Best CV RMSE: {-xgb_search.best_score_:.4f}")

    # Save all XGBoost parameter settings and their CV results
    xgb_cv_results = xgb_search.cv_results_
    xgb_param_results = pd.DataFrame(xgb_cv_results)
    xgb_param_results.to_csv(f"xgb_param_opt_corn_{_code}_{week_name}_all_params.csv", index=False)

    xgb = xgb_search.best_estimator_
    y_pred_train_xgb = xgb.predict(X_train)
    y_pred_test_xgb = xgb.predict(X_test)

    results.append({
        "week": week_name,
        "model": "XGBoost",
        "best_params": str(xgb_search.best_params_),
        "best_cv_rmse": -xgb_search.best_score_,
        "train_r2": r2_score(y_train, y_pred_train_xgb),
        "test_r2": r2_score(y_test, y_pred_test_xgb),
        "train_rmse": root_mean_squared_error(y_train, y_pred_train_xgb),
        "test_rmse": root_mean_squared_error(y_test, y_pred_test_xgb),
        "train_mae": mean_absolute_error(y_train, y_pred_train_xgb),
        "test_mae": mean_absolute_error(y_test, y_pred_test_xgb),
        "train_mape": mean_absolute_percentage_error(y_train, y_pred_train_xgb),
        "test_mape": mean_absolute_percentage_error(y_test, y_pred_test_xgb)
    })

    for fname, yt, yp in zip(test_labels, y_test, y_pred_test_xgb):
        file_level_results.append({
            "week": week_name, "model": "XGBoost", "file": fname,
            "y_true": yt, "y_pred": yp,
            "y_true_buacre": denormalize(yt),
            "y_pred_buacre": denormalize(yp),
        })

    # ============ PLSR Parameter Search ============
    print("\n" + "="*50)
    print("Searching for best PLSR parameters...")
    print("="*50)

    max_components = min(50, X_train.shape[1], X_train.shape[0])
    plsr_param_grid = {
        "n_components": list(range(5, max_components + 1, 5))
    }

    plsr_base = PLSRegression()

    plsr_search = GridSearchCV(
        estimator=plsr_base,
        param_grid=plsr_param_grid,
        scoring=rmse_scorer,
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    plsr_search.fit(X_train, y_train)

    print(f"\nBest PLSR parameters: {plsr_search.best_params_}")
    print(f"Best CV RMSE: {-plsr_search.best_score_:.4f}")

    # Save all PLSR parameter settings and their CV results
    plsr_cv_results = plsr_search.cv_results_
    plsr_param_results = pd.DataFrame({
        'n_components': plsr_cv_results['param_n_components'],
        'mean_test_rmse': -plsr_cv_results['mean_test_score'],
        'std_test_rmse': plsr_cv_results['std_test_score']
    })
    plsr_param_results.to_csv(f"plsr_param_opt_corn_{_code}_{week_name}_all_params.csv", index=False)

    plsr = plsr_search.best_estimator_
    y_pred_train_pls = plsr.predict(X_train).ravel()
    y_pred_test_pls = plsr.predict(X_test).ravel()

    results.append({
        "week": week_name,
        "model": "PLSR",
        "best_params": str(plsr_search.best_params_),
        "best_cv_rmse": -plsr_search.best_score_,
        "train_r2": r2_score(y_train, y_pred_train_pls),
        "test_r2": r2_score(y_test, y_pred_test_pls),
        "train_rmse": root_mean_squared_error(y_train, y_pred_train_pls),
        "test_rmse": root_mean_squared_error(y_test, y_pred_test_pls),
        "train_mae": mean_absolute_error(y_train, y_pred_train_pls),
        "test_mae": mean_absolute_error(y_test, y_pred_test_pls),
        "train_mape": mean_absolute_percentage_error(y_train, y_pred_train_pls),
        "test_mape": mean_absolute_percentage_error(y_test, y_pred_test_pls)
    })

    for fname, yt, yp in zip(test_labels, y_test, y_pred_test_pls):
        file_level_results.append({
            "week": week_name, "model": "PLSR", "file": fname,
            "y_true": yt, "y_pred": yp,
            "y_true_buacre": denormalize(yt),
            "y_pred_buacre": denormalize(yp),
        })

# Save results
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv(f"classical_ml_param_opt_corn_{_code}.csv", index=False)

df_file = pd.DataFrame(file_level_results)
df_file.to_csv(f"classical_ml_param_opt_field_level_corn_{_code}.csv", index=False)
print(f"\nSaved {len(df_results)} aggregate rows, {len(df_file)} field-level rows")
