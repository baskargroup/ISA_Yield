import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Configuration
DATA_ROOT = '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/conf_BS/w20_dataset/corn'
WEEK_NAME = 'processed_data_bs20_rm_8_19_7_17_12_9_10_2_1'
biweek_folder = os.path.join(DATA_ROOT, WEEK_NAME)
modalities = ["S2L2A", "S1GRD", "CDL", "DEM", "WEATHER"]
STATIC_MODALITIES = {"CDL", "DEM"}
label_folder = "yield_geotiffs"
_code = "s12cdw"
CROP = "corn"

# Split files
splits = {}
for split in ["train", "val", "test"]:
    split_file = os.path.join(biweek_folder, f"{split}_{CROP}.txt")
    if not os.path.exists(split_file):
        split_file = os.path.join(biweek_folder, split, f"{split}_{CROP}.txt")
    with open(split_file) as f:
        splits[split] = [line.strip() + ".npy" for line in f if line.strip()]

modal_paths = {m: os.path.join(biweek_folder, m) for m in modalities}
label_path = os.path.join(biweek_folder, label_folder)

# Per-channel normalization statistics (z-score)
NORM_STATS = {
    "S2L2A": {
        "mean": np.array([1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975, 2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648]),
        "std": np.array([2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311]),
    },
    "S1GRD": {
        "mean": np.array([-12.599, -20.293]),
        "std": np.array([5.195, 5.890]),
    },
    "DEM": {
        "mean": np.array([670.665]),
        "std": np.array([951.272]),
    },
    "WEATHER": {
        "mean": np.array([23.818, 24.606, 12.064, 373.494, 1482.514, 0.547, 49955.316]),
        "std": np.array([7.991, 1.232, 1.241, 11.317, 120.417, 1.337, 504.205]),
    },
    # CDL is categorical — not normalized
}

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
            if m in STATIC_MODALITIES:
                arr = arr[0:1]
            if m in NORM_STATS:
                ns = NORM_STATS[m]
                arr = (arr - ns["mean"].reshape(1, -1, 1, 1)) / ns["std"].reshape(1, -1, 1, 1)
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
        mask = np.isfinite(y) & (y > -1)
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
        row['file'] = os.path.splitext(fname)[0]
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
X_val, y_val, val_labels = build_X_y(splits["val"], modal_paths, label_path)
X_test, y_test, test_labels = build_X_y(splits["test"], modal_paths, label_path)
print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Val:   X={X_val.shape}, y={y_val.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

# Standardize: fit on train only
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# PredefinedSplit: train=-1 (always train), val=0 (validation fold)
X_trainval = np.vstack([X_train_s, X_val_s])
y_trainval = np.concatenate([y_train, y_val])
test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
ps = PredefinedSplit(test_fold)

rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

# PLSR Parameter Search
max_components = min(50, X_train.shape[1], X_train.shape[0])
plsr_param_grid = {"n_components": list(range(1, max_components + 1))}
plsr_base = PLSRegression()
plsr_search = GridSearchCV(
    estimator=plsr_base,
    param_grid=plsr_param_grid,
    scoring=rmse_scorer,
    cv=ps,
    verbose=2,
    n_jobs=-1
)
plsr_search.fit(X_trainval, y_trainval)
print(f"\nBest PLSR parameters: {plsr_search.best_params_}")
print(f"Best val RMSE: {-plsr_search.best_score_:.4f}")

# Save all parameter settings and their CV results
cv_results = plsr_search.cv_results_
param_results = pd.DataFrame({
    'n_components': cv_results['param_n_components'],
    'mean_test_rmse': -cv_results['mean_test_score'],
    'std_test_rmse': cv_results['std_test_score']
})
param_results.to_csv(f"classical_ml_param_opt_{CROP}_{_code}_plsr_all_params.csv", index=False)

plsr = plsr_search.best_estimator_
y_pred_train = plsr.predict(X_train_s).ravel()
y_pred_val = plsr.predict(X_val_s).ravel()
y_pred_test = plsr.predict(X_test_s).ravel()

results = {
    "model": "PLSR",
    "week": WEEK_NAME,
    "best_params": str(plsr_search.best_params_),
    "best_val_rmse": -plsr_search.best_score_,
    "train_r2": r2_score(y_train, y_pred_train),
    "val_r2": r2_score(y_val, y_pred_val),
    "test_r2": r2_score(y_test, y_pred_test),
    "train_rmse": root_mean_squared_error(y_train, y_pred_train),
    "val_rmse": root_mean_squared_error(y_val, y_pred_val),
    "test_rmse": root_mean_squared_error(y_test, y_pred_test),
    "train_mae": mean_absolute_error(y_train, y_pred_train),
    "val_mae": mean_absolute_error(y_val, y_pred_val),
    "test_mae": mean_absolute_error(y_test, y_pred_test),
    "train_mape": mean_absolute_percentage_error(y_train, y_pred_train),
    "val_mape": mean_absolute_percentage_error(y_val, y_pred_val),
    "test_mape": mean_absolute_percentage_error(y_test, y_pred_test)
}
print("\nPLSR Results:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

df_results = pd.DataFrame([results])
df_results.to_csv(f"classical_ml_param_opt_{CROP}_{_code}_plsr.csv", index=False)

# =====================================================
# ROUND 2: Vegetation Index Features Only
# =====================================================

OPTICAL_VI_NAMES = [
    'GLI', 'NGRDI', 'VARI', 'VEG', 'RGBVI', 'ExG', 'ExR',
    'NDVI', 'GNDVI', 'EVI', 'SAVI', 'NDRE', 'RDVI',
    'CIgreen', 'CIrededge', 'OSAVI', 'MSAVI', 'NDI45', 'MTCI',
    'NDWI', 'NBR', 'NDMI', 'SR', 'PSRI'
]
SAR_VI_NAMES = ['RVI', 'VH_VV', 'NRPB']
ALL_VI_NAMES = OPTICAL_VI_NAMES + SAR_VI_NAMES


def compute_optical_vis(s2_arr):
    """Compute 24 vegetation indices from S2L2A bands.
    Input: s2_arr (T, 12, H, W) - DN values scaled by 10000
    Bands: [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]
    """
    eps = 1e-8
    s2 = np.nan_to_num(s2_arr.astype(np.float32), nan=0.0) / 10000.0

    B = s2[:, 1]     # B02 Blue
    G = s2[:, 2]     # B03 Green
    R = s2[:, 3]     # B04 Red
    RE1 = s2[:, 4]   # B05 Red Edge 1
    RE2 = s2[:, 5]   # B06 Red Edge 2
    NIR = s2[:, 7]   # B08 NIR
    B8A = s2[:, 8]   # B8A NIR narrow
    SWIR1 = s2[:, 10] # B11 SWIR 1
    SWIR2 = s2[:, 11] # B12 SWIR 2

    # RGB-based (7)
    GLI = (2*G - R - B) / (2*G + R + B + eps)
    NGRDI = (G - R) / (G + R + eps)
    VARI = (G - R) / (G + R - B + eps)
    VEG = G / (np.power(np.maximum(R, eps), 0.667) * np.power(np.maximum(B, eps), 0.333) + eps)
    RGBVI = (G**2 - R*B) / (G**2 + R*B + eps)
    ExG = 2*G - R - B
    ExR = 1.4*R - G

    # NIR-based (6)
    NDVI = (NIR - R) / (NIR + R + eps)
    GNDVI = (NIR - G) / (NIR + G + eps)
    EVI_val = 2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1 + eps)
    SAVI = 1.5 * (NIR - R) / (NIR + R + 0.5 + eps)
    NDRE = (NIR - RE1) / (NIR + RE1 + eps)
    RDVI = (NIR - R) / (np.sqrt(np.maximum(NIR + R, 0)) + eps)

    # Additional Red Edge / NIR (6)
    CIgreen = NIR / (G + eps) - 1
    CIrededge = NIR / (RE1 + eps) - 1
    OSAVI = 1.16 * (NIR - R) / (NIR + R + 0.16 + eps)
    MSAVI = (2*NIR + 1 - np.sqrt(np.maximum((2*NIR + 1)**2 - 8*(NIR - R), 0))) / 2
    NDI45 = (RE1 - R) / (RE1 + R + eps)
    MTCI = (RE2 - RE1) / (RE1 - R + eps)

    # Moisture / SWIR (3)
    NDWI = (NIR - SWIR1) / (NIR + SWIR1 + eps)
    NBR = (NIR - SWIR2) / (NIR + SWIR2 + eps)
    NDMI = (B8A - SWIR1) / (B8A + SWIR1 + eps)

    # Other (2)
    SR = NIR / (R + eps)
    PSRI = (R - G) / (RE1 + eps)

    vis = np.stack([GLI, NGRDI, VARI, VEG, RGBVI, ExG, ExR,
                    NDVI, GNDVI, EVI_val, SAVI, NDRE, RDVI,
                    CIgreen, CIrededge, OSAVI, MSAVI, NDI45, MTCI,
                    NDWI, NBR, NDMI, SR, PSRI], axis=1)
    vis = np.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
    vis = np.clip(vis, -5, 5)
    return vis


def compute_sar_vis(s1_arr):
    """Compute 3 SAR vegetation indices from S1GRD bands.
    Input: s1_arr (T, 2, H, W) - dB values, band 0=VV, band 1=VH
    """
    eps = 1e-8
    s1 = np.nan_to_num(s1_arr.astype(np.float32), nan=-30.0)
    # Convert dB to linear scale
    vv = np.power(10.0, s1[:, 0] / 10.0)
    vh = np.power(10.0, s1[:, 1] / 10.0)

    RVI = 4 * vh / (vv + vh + eps)
    VH_VV = vh / (vv + eps)
    NRPB = (vh - vv) / (vh + vv + eps)

    vis = np.stack([RVI, VH_VV, NRPB], axis=1)
    vis = np.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
    vis = np.clip(vis, -5, 5)
    return vis


def build_X_y_vi(file_names, s2_path, s1_path, label_path):
    """Build features using only vegetation indices (optical + SAR)."""
    aggregated_rows = []
    for fname in tqdm(file_names, desc="Files (VI)"):
        s2_file = os.path.join(s2_path, fname)
        s1_file = os.path.join(s1_path, fname)
        if not os.path.exists(s2_file) or not os.path.exists(s1_file):
            continue

        optical_vis = compute_optical_vis(np.load(s2_file))
        sar_vis = compute_sar_vis(np.load(s1_file))
        all_vis = np.concatenate([optical_vis, sar_vis], axis=1)

        arr = all_vis.reshape(all_vis.shape[0], all_vis.shape[1], -1)
        arr = arr.transpose(2, 0, 1).reshape(-1, all_vis.shape[0]*all_vis.shape[1])

        y = np.load(os.path.join(label_path, fname))
        if y.ndim == 3:
            y = y.squeeze(0)
        y = y.flatten()
        mask = np.isfinite(y) & (y > -1)
        X_filtered = arr[mask]
        y_filtered = y[mask]
        if len(y_filtered) == 0:
            continue
        X_mean = np.nanmean(X_filtered, axis=0)
        y_mean = np.nanmean(y_filtered)
        if np.any(np.isnan(X_mean)) or np.isnan(y_mean):
            continue
        row = {f'feature_{j}': X_mean[j] for j in range(len(X_mean))}
        row['yield'] = y_mean
        row['file'] = os.path.splitext(fname)[0]
        aggregated_rows.append(row)
    df_agg = pd.DataFrame(aggregated_rows)
    if len(df_agg) == 0:
        return np.empty((0, 1)), np.empty((0,)), np.empty((0,), dtype=object)
    feature_cols = [col for col in df_agg.columns if col.startswith('feature_')]
    X = df_agg[feature_cols].values
    y = df_agg['yield'].values
    file_labels = df_agg['file'].values
    return X, y, file_labels


print("\n" + "="*60)
print("ROUND 2: Vegetation Index Features Only")
print(f"Optical VIs ({len(OPTICAL_VI_NAMES)}): {OPTICAL_VI_NAMES}")
print(f"SAR VIs ({len(SAR_VI_NAMES)}): {SAR_VI_NAMES}")
print("="*60)

s2_path = modal_paths["S2L2A"]
s1_path = modal_paths["S1GRD"]

X_train_vi, y_train_vi, _ = build_X_y_vi(splits["train"], s2_path, s1_path, label_path)
X_val_vi, y_val_vi, _ = build_X_y_vi(splits["val"], s2_path, s1_path, label_path)
X_test_vi, y_test_vi, _ = build_X_y_vi(splits["test"], s2_path, s1_path, label_path)
print(f"Train VI: X={X_train_vi.shape}, y={y_train_vi.shape}")
print(f"Val VI:   X={X_val_vi.shape}, y={y_val_vi.shape}")
print(f"Test VI:  X={X_test_vi.shape}, y={y_test_vi.shape}")

# Standardize VI features
scaler_vi = StandardScaler()
X_train_vi_s = scaler_vi.fit_transform(X_train_vi)
X_val_vi_s = scaler_vi.transform(X_val_vi)
X_test_vi_s = scaler_vi.transform(X_test_vi)

X_trainval_vi = np.vstack([X_train_vi_s, X_val_vi_s])
y_trainval_vi = np.concatenate([y_train_vi, y_val_vi])
test_fold_vi = np.concatenate([np.full(len(X_train_vi), -1), np.full(len(X_val_vi), 0)])
ps_vi = PredefinedSplit(test_fold_vi)

# PLSR Parameter Search with VI features
max_components_vi = min(50, X_train_vi.shape[1], X_train_vi.shape[0])
plsr_param_grid_vi = {"n_components": list(range(1, max_components_vi + 1))}
plsr_base_vi = PLSRegression()
plsr_search_vi = GridSearchCV(
    estimator=plsr_base_vi,
    param_grid=plsr_param_grid_vi,
    scoring=rmse_scorer,
    cv=ps_vi,
    verbose=2,
    n_jobs=-1
)
plsr_search_vi.fit(X_trainval_vi, y_trainval_vi)
print(f"\nBest PLSR VI parameters: {plsr_search_vi.best_params_}")
print(f"Best val RMSE (VI): {-plsr_search_vi.best_score_:.4f}")

cv_results_vi = plsr_search_vi.cv_results_
param_results_vi = pd.DataFrame({
    'n_components': cv_results_vi['param_n_components'],
    'mean_test_rmse': -cv_results_vi['mean_test_score'],
    'std_test_rmse': cv_results_vi['std_test_score']
})
param_results_vi.to_csv(f"classical_ml_param_opt_{CROP}_{_code}_plsr_vi_all_params.csv", index=False)

plsr_vi = plsr_search_vi.best_estimator_
y_pred_train_vi = plsr_vi.predict(X_train_vi_s).ravel()
y_pred_val_vi = plsr_vi.predict(X_val_vi_s).ravel()
y_pred_test_vi = plsr_vi.predict(X_test_vi_s).ravel()

results_vi = {
    "model": "PLSR_VI",
    "week": WEEK_NAME,
    "best_params": str(plsr_search_vi.best_params_),
    "best_val_rmse": -plsr_search_vi.best_score_,
    "train_r2": r2_score(y_train_vi, y_pred_train_vi),
    "val_r2": r2_score(y_val_vi, y_pred_val_vi),
    "test_r2": r2_score(y_test_vi, y_pred_test_vi),
    "train_rmse": root_mean_squared_error(y_train_vi, y_pred_train_vi),
    "val_rmse": root_mean_squared_error(y_val_vi, y_pred_val_vi),
    "test_rmse": root_mean_squared_error(y_test_vi, y_pred_test_vi),
    "train_mae": mean_absolute_error(y_train_vi, y_pred_train_vi),
    "val_mae": mean_absolute_error(y_val_vi, y_pred_val_vi),
    "test_mae": mean_absolute_error(y_test_vi, y_pred_test_vi),
    "train_mape": mean_absolute_percentage_error(y_train_vi, y_pred_train_vi),
    "val_mape": mean_absolute_percentage_error(y_val_vi, y_pred_val_vi),
    "test_mape": mean_absolute_percentage_error(y_test_vi, y_pred_test_vi)
}
print("\nPLSR VI Results:")
for key, value in results_vi.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

df_results_vi = pd.DataFrame([results_vi])
df_results_vi.to_csv(f"classical_ml_param_opt_{CROP}_{_code}_plsr_vi.csv", index=False)
