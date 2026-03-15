# Cell 2: Train and test for a single biweek
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
# Cell 1: Imports
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

# Configuration
BIWEEK_NUMBER = 24  # Specify which biweek to process
base_dir = '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/processed_data/weekly_24'
biweek_folder = os.path.join(base_dir, f"processed_data_biweekly_{BIWEEK_NUMBER}")
_code = "s12wd"
label_folder = "yield_geotiffs"

modalities = ["S2L2A", 
              "S1GRD",
              "DEM", 
              "WEATHER"]

def read_split(split_path):
    """Read split file and return list of .npy filenames"""
    with open(split_path) as f:
        return [line.strip() + ".npy" for line in f if line.strip()]

def build_X_y(file_names, modal_paths, label_path):
    """Build aggregated feature matrix and labels at file level"""
    aggregated_rows = []
    
    for fname in tqdm(file_names, desc="Files"):
        feats = []
        skip = False
        
        # Load all modalities
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
        
        # Concatenate features
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
        
        # Aggregate this file - compute mean for all features and yield
        X_mean = X_filtered.mean(axis=0)
        y_mean = y_filtered.mean()
        
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

print(f"Processing biweek: {BIWEEK_NUMBER}")

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
print("Building training data...")
X_train, y_train, train_labels = build_X_y(splits["train"], modal_paths, label_path)
print(f"Train shape: X={X_train.shape}, y={y_train.shape}")

print("Building test data...")
X_test, y_test, test_labels = build_X_y(splits["test"], modal_paths, label_path)
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

# Define scoring metric for cross-validation
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

# ============ XGBoost Parameter Search ============
print("\n" + "="*50)
print("Searching for best XGBoost parameters...")
print("="*50)

# Define parameter grid for XGBoost
xgb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha": [0.0, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
}

# Use RandomizedSearchCV for efficiency (GridSearchCV would take too long)
xgb_base = XGBRegressor(
    n_jobs=-1,
    random_state=42,
    tree_method="hist"
)

xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_grid,
    n_iter=50,  # Number of parameter settings sampled
    scoring=rmse_scorer,
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)

print("\nBest XGBoost parameters found:")
print(xgb_search.best_params_)
print(f"Best CV RMSE: {-xgb_search.best_score_:.4f}")

# Save all XGBoost parameter settings and their CV results
xgb_cv_results = xgb_search.cv_results_
xgb_param_results = pd.DataFrame(xgb_cv_results)
xgb_param_results.to_csv(f"xgb_param_opt_all_params_biweek{BIWEEK_NUMBER}.csv", index=False)

# Train final model with best parameters
print("\nTraining XGBoost with best parameters...")
xgb = xgb_search.best_estimator_

y_pred_train_xgb = xgb.predict(X_train)
y_pred_test_xgb = xgb.predict(X_test)

xgb_results = {
    "model": "XGBoost",
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
for key, value in xgb_results.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# ============ PLSR Parameter Search ============
print("\n" + "="*50)
print("Searching for best PLSR parameters...")
print("="*50)

# Define parameter grid for PLSR
# Test different numbers of components
max_components = min(50, X_train.shape[1], X_train.shape[0])
plsr_param_grid = {
    "n_components": list(range(5, max_components + 1, 5))  # Test every 5th component from 5 to max
}

plsr_base = PLSRegression()

plsr_search = GridSearchCV(
    estimator=plsr_base,
    param_grid=plsr_param_grid,
    scoring=rmse_scorer,
    cv=5,  # 5-fold cross-validation
    verbose=2,
    n_jobs=-1
)

plsr_search.fit(X_train, y_train)

print("\nBest PLSR parameters found:")
print(plsr_search.best_params_)
print(f"Best CV RMSE: {-plsr_search.best_score_:.4f}")

# Save all PLSR parameter settings and their CV results
plsr_cv_results = plsr_search.cv_results_
plsr_param_results = pd.DataFrame({
    'n_components': plsr_cv_results['param_n_components'],
    'mean_test_rmse': -plsr_cv_results['mean_test_score'],
    'std_test_rmse': plsr_cv_results['std_test_score']
})
plsr_param_results.to_csv(f"plsr_param_opt_all_params_biweek{BIWEEK_NUMBER}.csv", index=False)

# Train final model with best parameters
print("\nTraining PLSR with best parameters...")
plsr = plsr_search.best_estimator_

y_pred_train_pls = plsr.predict(X_train).ravel()
y_pred_test_pls = plsr.predict(X_test).ravel()

plsr_results = {
    "model": "PLSR",
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
for key, value in plsr_results.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# Save file-level predictions
df_train_pred = pd.DataFrame({
    'file': train_labels,
    'y_true': y_train,
    'y_pred_xgb': y_pred_train_xgb,
    'y_pred_plsr': y_pred_train_pls
})

df_test_pred = pd.DataFrame({
    'file': test_labels,
    'y_true': y_test,
    'y_pred_xgb': y_pred_test_xgb,
    'y_pred_plsr': y_pred_test_pls
})

# Save results
output_file = f"classical_ml_file_level_results_{_code}_biweek{BIWEEK_NUMBER}.csv"
df_all = pd.concat([
    df_train_pred.assign(split='train'),
    df_test_pred.assign(split='test')
], ignore_index=True)
df_all.to_csv(output_file, index=False)
print(f"\nFile-level predictions saved to: {output_file}")

# Summary dataframe
df_summary = pd.DataFrame([xgb_results, plsr_results])
df_summary['biweek'] = BIWEEK_NUMBER

# Save best parameters
df_summary['best_params'] = [str(xgb_search.best_params_), str(plsr_search.best_params_)]
df_summary['best_cv_rmse'] = [-xgb_search.best_score_, -plsr_search.best_score_]

print("\nSummary:")
print(df_summary)

# Save summary with best parameters
summary_file = f"classical_ml_summary_{_code}_biweek{BIWEEK_NUMBER}.csv"
df_summary.to_csv(summary_file, index=False)
print(f"\nSummary with best parameters saved to: {summary_file}")
