import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pandas as pd

# Configuration
DATA_ROOT = 'processed_data/weekly_24'
WEEK_NAME = 'processed_data_weekly_13'
biweek_folder = os.path.join(DATA_ROOT, WEEK_NAME)
modalities = ["S2L2A", "S1GRD", "DEM", "SOIL"]
label_folder = "yield_geotiffs"
_code = "s12ds"

# Split files
splits = {}
for split in ["train", "val", "test"]:
    split_file = os.path.join(biweek_folder, f"{split}_soybean.txt")
    if not os.path.exists(split_file):
        split_file = os.path.join(biweek_folder, split, f"{split}_soybean.txt")
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

# PLSR Parameter Search
max_components = min(50, X_train.shape[1], X_train.shape[0])
plsr_param_grid = {"n_components": list(range(5, max_components + 1, 5))}
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

# Save all parameter settings and their CV results
cv_results = plsr_search.cv_results_
param_results = pd.DataFrame({
    'n_components': cv_results['param_n_components'],
    'mean_test_rmse': -cv_results['mean_test_score'],
    'std_test_rmse': cv_results['std_test_score']
})
param_results.to_csv(f"classical_ml_param_opt_soybean_{_code}_weekly13_plsr_all_params.csv", index=False)
plsr = plsr_search.best_estimator_
y_pred_train_pls = plsr.predict(X_train).ravel()
y_pred_test_pls = plsr.predict(X_test).ravel()
results = {
    "model": "PLSR",
    "week": WEEK_NAME,
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
}
print("\nPLSR Results:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
df_results = pd.DataFrame([results])
df_results.to_csv(f"classical_ml_param_opt_soybean_{_code}_weekly13_plsr.csv", index=False)
