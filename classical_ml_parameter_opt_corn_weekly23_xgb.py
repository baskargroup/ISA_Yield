import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import pandas as pd

# Configuration
DATA_ROOT = 'processed_data/weekly_24'
WEEK_NAME = 'processed_data_weekly_23'
biweek_folder = os.path.join(DATA_ROOT, WEEK_NAME)
modalities = ["S2L2A", "S1GRD", "CDL", "DEM", "WEATHER"]
label_folder = "yield_geotiffs"
_code = "s12cdw"

# Split files
splits = {}
for split in ["train", "val", "test"]:
    split_file = os.path.join(biweek_folder, f"{split}_corn.txt")
    if not os.path.exists(split_file):
        split_file = os.path.join(biweek_folder, split, f"{split}_corn.txt")
    splits[split] = [line.strip() + ".npy" for line in open(split_file) if line.strip()]

modal_paths = {m: os.path.join(biweek_folder, m) for m in modalities}
label_path = os.path.join(biweek_folder, label_folder)

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

print(f"Processing week: {WEEK_NAME}")
X_train, y_train, train_labels = build_X_y(splits["train"], modal_paths, label_path)
X_test, y_test, test_labels = build_X_y(splits["test"], modal_paths, label_path)
print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

# XGBoost Parameter Search
xgb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha": [0.0, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
}
xgb_base = XGBRegressor(n_jobs=-1, random_state=42, tree_method="hist")
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

# Save all parameter settings and their CV results
xgb_cv_results = xgb_search.cv_results_
xgb_param_results = pd.DataFrame(xgb_cv_results)
xgb_param_results.to_csv(f"classical_ml_param_opt_corn_{_code}_weekly23_xgb_all_params.csv", index=False)
xgb = xgb_search.best_estimator_
y_pred_train_xgb = xgb.predict(X_train)
y_pred_test_xgb = xgb.predict(X_test)
results = {
    "model": "XGBoost",
    "week": WEEK_NAME,
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
}
print("\nXGBoost Results:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
df_results = pd.DataFrame([results])
df_results.to_csv(f"classical_ml_param_opt_corn_{_code}_weekly23_xgb.csv", index=False)
