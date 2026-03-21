#!/usr/bin/env python3
"""
Diebold-Mariano (DM) test: Best Classical ML (XGBoost, optimised params)
vs Best TerraMind M3 configuration, per crop.

Strategy:
  1. Load M3 predictions (already in bu/acre) to get the target field set.
  2. Load classical ML field-level predictions from saved CSVs produced by
     `classical_ml_parameter_opt_{corn,soybean}.py`.  If the CSV contains
     `y_pred_buacre` (denormalised), use it directly.  Otherwise denormalise
     the normalised `y_pred` column.
  3. If no saved CSV is found, fall back to re-training XGBoost from scratch.
  4. Use M3's YieldGT (bu/acre) as the common ground truth for both models.
  5. Run HLN-corrected DM test on paired squared errors.

Normalisation (process_ts_weekly_chloe.py):
  Corn:    norm = (raw - 50) / (370 - 50)   →  denorm: raw = norm * 320 + 50
  Soybean: norm = (raw - 30) / (120 - 30)   →  denorm: raw = norm * 90  + 30
  Values clipped: >1 → 1,  <0 → -1  (where -1 = invalid / masked)

Loss metric:  Squared error  (positive DM → M3 better; negative → CML better)

Test variant: Harvey-Leybourne-Newbold (1997) small-sample-corrected DM,
              h=1, two-tailed t-distribution p-value.

Output:
  dm_test_results.csv
"""

import os
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor


# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT = 'processed_data/weekly_24'

# Denormalisation constants  (from process_ts_weekly_chloe.py)
DENORM = {
    'corn':    {'data_min': 50.0, 'data_max': 370.0},   # range = 320
    'soybean': {'data_min': 30.0, 'data_max': 120.0},   # range = 90
}

CORN_CONFIG = dict(
    crop='Corn',
    week='processed_data_weekly_23',
    modalities=['S2L2A', 'S1GRD', 'CDL', 'DEM', 'WEATHER'],
    split_crop_key='corn',
    best_params_csv='classical_ml_param_opt_corn_s12cdw_weekly23_xgb.csv',
    field_level_csv='classical_ml_param_opt_field_level_corn_s12cdw.csv',
)

SOY_CONFIG = dict(
    crop='Soybean',
    week='processed_data_weekly_21',
    modalities=['S2L2A', 'S1GRD', 'DEM', 'SOIL'],
    split_crop_key='soybean',
    best_params_csv='classical_ml_param_opt_soybean_s12ds_weekly21_xgb.csv',
    field_level_csv='classical_ml_param_opt_field_level_soybean_s12ds.csv',
)

M3_PRED_DIR = 'M3/predictions'
OUTPUT_CSV = 'dm_test_results.csv'


# ============================================================================
# Helpers
# ============================================================================

def denormalize(y_norm, crop_key):
    """Convert normalised [0,1] yields back to bu/acre."""
    d = DENORM[crop_key]
    return y_norm * (d['data_max'] - d['data_min']) + d['data_min']


def build_X_y(file_names, modal_paths, label_path, modalities):
    """Load pixel-averaged features + normalised yields for a list of fields."""
    rows = []
    for fname in tqdm(file_names, desc='  Loading', leave=False):
        feats = []
        skip = False
        for m in modalities:
            mfile = os.path.join(modal_paths[m], fname)
            if not os.path.exists(mfile):
                skip = True
                break
            arr = np.load(mfile)
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
            arr = arr.transpose(2, 0, 1).reshape(-1, arr.shape[0] * arr.shape[1])
            feats.append(arr)
        if skip:
            continue
        X = np.concatenate(feats, axis=1)
        y = np.load(os.path.join(label_path, fname))
        if y.ndim == 3:
            y = y.squeeze(0)
        y = y.flatten()
        # Mask: keep finite, non-zero pixels (consistent with CML param opt)
        mask = np.isfinite(y) & (y != 0)
        X_f = X[mask]; y_f = y[mask]
        if len(y_f) == 0:
            continue
        X_mean = np.nanmean(X_f, axis=0)
        y_mean = np.nanmean(y_f)
        if np.any(np.isnan(X_mean)) or np.isnan(y_mean):
            continue
        rows.append({'file': os.path.splitext(fname)[0],
                     'yield': y_mean,
                     **{f'feat_{j}': X_mean[j] for j in range(len(X_mean))}})
    df = pd.DataFrame(rows)
    if df.empty:
        return np.empty((0, 1)), np.empty((0,)), np.array([], dtype=str)
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    return df[feat_cols].values, df['yield'].values, df['file'].values


def build_X_only(file_names, modal_paths, modalities):
    """Load pixel-averaged features only (no yield) for target fields."""
    rows = []
    for fname in tqdm(file_names, desc='  Loading features', leave=False):
        feats = []
        skip = False
        for m in modalities:
            mfile = os.path.join(modal_paths[m], fname)
            if not os.path.exists(mfile):
                skip = True
                break
            arr = np.load(mfile)
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
            arr = arr.transpose(2, 0, 1).reshape(-1, arr.shape[0] * arr.shape[1])
            feats.append(arr)
        if skip:
            continue
        X = np.concatenate(feats, axis=1)
        X_mean = np.nanmean(X, axis=0)
        if np.any(np.isnan(X_mean)):
            continue
        rows.append({'file': os.path.splitext(fname)[0],
                     **{f'feat_{j}': X_mean[j] for j in range(len(X_mean))}})
    df = pd.DataFrame(rows)
    if df.empty:
        return np.empty((0, 1)), np.array([], dtype=str)
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    return df[feat_cols].values, df['file'].values


# ============================================================================
# Classical ML: load saved predictions or retrain
# ============================================================================

def load_classical_predictions(cfg):
    """
    Load field-level XGBoost predictions (bu/acre) from the saved CSV produced
    by classical_ml_parameter_opt_{crop}.py.  Filters for the best week and
    model='XGBoost'.

    Returns
    -------
    pd.DataFrame  with columns: file, y_pred_cml_buacre
    None          if the CSV does not exist
    """
    csv_path = cfg['field_level_csv']
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    # Filter to best week + XGBoost
    week = cfg['week']
    mask = (df['week'] == week) & (df['model'] == 'XGBoost')
    sub = df.loc[mask].copy()
    if sub.empty:
        return None

    key = cfg['split_crop_key']
    if 'y_pred_buacre' in sub.columns:
        sub = sub.rename(columns={'y_pred_buacre': 'y_pred_cml_buacre'})
    else:
        # Denormalise from normalised y_pred
        sub['y_pred_cml_buacre'] = denormalize(sub['y_pred'].values, key)

    return sub[['file', 'y_pred_cml_buacre']].reset_index(drop=True)


def get_classical_predictions(cfg, target_field_names):
    """
    Train XGBoost with saved best params on the current train split
    (normalised data), then predict on *target_field_names* and denormalise
    predictions to bu/acre.

    Parameters
    ----------
    cfg : dict              crop configuration
    target_field_names : list[str]   field names (no extension) to predict on

    Returns
    -------
    pd.DataFrame  with columns: file, y_pred_cml_buacre
    """
    week_dir = os.path.join(DATA_ROOT, cfg['week'])
    modalities = cfg['modalities']
    key = cfg['split_crop_key']

    # --- Load train split ---
    train_path = os.path.join(week_dir, f'train_{key}.txt')
    if not os.path.exists(train_path):
        train_path = os.path.join(week_dir, 'train', f'train_{key}.txt')
    with open(train_path) as f:
        train_files = [l.strip() + '.npy' for l in f if l.strip()]

    modal_paths = {m: os.path.join(week_dir, m) for m in modalities}
    label_path = os.path.join(week_dir, 'yield_geotiffs')

    # --- Build training data ---
    print(f'  Building train set ({cfg["crop"]})…')
    X_train, y_train, _ = build_X_y(train_files, modal_paths, label_path, modalities)
    print(f'  Train samples: {len(y_train)}')

    # --- Build features for M3 target fields ---
    target_npy = [fn + '.npy' for fn in target_field_names]
    print(f'  Building features for {len(target_npy)} M3 target fields…')
    X_target, target_names = build_X_only(target_npy, modal_paths, modalities)
    print(f'  Target fields with valid features: {len(target_names)}')

    if len(X_train) == 0 or len(X_target) == 0:
        return pd.DataFrame(columns=['file', 'y_pred_cml_buacre'])

    # --- Train ---
    row = pd.read_csv(cfg['best_params_csv']).iloc[0]
    raw = row['best_params']
    params = ast.literal_eval(raw) if isinstance(raw, str) else {}

    print(f'  Training XGBoost with best params: {params}')
    model = XGBRegressor(n_jobs=-1, random_state=42, tree_method='hist', **params)
    model.fit(X_train, y_train)

    # --- Predict (normalised) → denormalise to bu/acre ---
    y_pred_norm = model.predict(X_target)
    y_pred_buacre = denormalize(y_pred_norm, key)

    return pd.DataFrame({'file': target_names, 'y_pred_cml_buacre': y_pred_buacre})


# ============================================================================
# Best M3 predictions for a crop
# ============================================================================

def get_best_m3_predictions(crop):
    """Return (DataFrame, modal_name, R²) for the best M3 config for a crop."""
    pred_dir = Path(M3_PRED_DIR)
    best_r2 = -np.inf
    best_df = None
    best_modal = None
    for csv_f in sorted(pred_dir.glob(f'*_{crop}.csv')):
        df = pd.read_csv(csv_f)
        r2 = r2_score(df['YieldGT'].values, df['Prediction'].values)
        if r2 > best_r2:
            best_r2 = r2
            best_df = df.rename(columns={'Filename': 'file',
                                         'YieldGT': 'y_true_buacre',
                                         'Prediction': 'y_pred_m3_buacre'})
            best_modal = csv_f.stem
    return best_df, best_modal, best_r2


# ============================================================================
# DM test (Harvey-Leybourne-Newbold corrected)
# ============================================================================

def dm_test_hlb(e1, e2, h=1):
    """
    Two-tailed Harvey-Leybourne-Newbold (1997) corrected DM test.

    e1, e2 : 1-D arrays of forecast errors (y_pred - y_true) for model 1 & 2.
    h      : forecast horizon (1 for single-step).

    Returns
    -------
    dm_stat : float   HLN-corrected DM statistic
    p_value : float   two-tailed p-value (t distribution, df=T-1)
    d_bar   : float   mean loss differential
    """
    d = e1 ** 2 - e2 ** 2          # loss differential (MSE-based)
    T = len(d)
    d_bar = np.mean(d)

    # Newey-West variance estimate (truncated at h-1 lags)
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += gamma_k
    var_d = (gamma0 + 2 * gamma_sum) / T

    if var_d <= 0:
        return np.nan, np.nan, d_bar

    dm_raw = d_bar / np.sqrt(var_d)

    # HLN small-sample correction factor
    hlb_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat = dm_raw * hlb_factor

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=T - 1))
    return dm_stat, p_value, d_bar


# ============================================================================
# Main
# ============================================================================

def run_dm_tests():
    records = []

    for cfg in [CORN_CONFIG, SOY_CONFIG]:
        crop = cfg['crop']
        print(f'\n{"="*60}')
        print(f'Crop: {crop}')
        print('='*60)

        # --- Best M3 predictions (already bu/acre) ---
        m3_df, m3_modal, m3_r2 = get_best_m3_predictions(crop)
        if m3_df is None:
            print(f'  No M3 predictions found for {crop}, skipping.')
            continue
        print(f'  M3 best config : {m3_modal}  (R²={m3_r2:.4f})')
        print(f'  M3 test fields : {len(m3_df)}')

        # --- Classical ML: load saved predictions or retrain ---
        m3_field_names = m3_df['file'].tolist()
        cml_df = load_classical_predictions(cfg)
        if cml_df is not None:
            print(f'  Loaded {len(cml_df)} CML predictions from {cfg["field_level_csv"]}')
        else:
            print(f'  No saved predictions found, re-training from scratch…')
            cml_df = get_classical_predictions(cfg, m3_field_names)
        print(f'  CML predicted fields: {len(cml_df)}')

        # --- Merge on shared field names ---
        merged = pd.merge(cml_df, m3_df, on='file', how='inner')
        n_paired = len(merged)
        print(f'  Paired fields after inner join: {n_paired}')
        if n_paired < 5:
            print('  Too few paired fields for reliable DM test, skipping.')
            continue

        # Common ground truth from M3 (bu/acre)
        y_true   = merged['y_true_buacre'].values
        y_pred_c = merged['y_pred_cml_buacre'].values
        y_pred_m = merged['y_pred_m3_buacre'].values

        e_cml = y_pred_c - y_true
        e_m3  = y_pred_m - y_true

        # Metrics on paired subset (all in bu/acre)
        r2_cml  = r2_score(y_true, y_pred_c)
        r2_m3   = r2_score(y_true, y_pred_m)
        mae_cml = mean_absolute_error(y_true, y_pred_c)
        mae_m3  = mean_absolute_error(y_true, y_pred_m)
        rmse_cml = root_mean_squared_error(y_true, y_pred_c)
        rmse_m3  = root_mean_squared_error(y_true, y_pred_m)

        # DM test: e1 = classical, e2 = M3
        # Positive DM → classical has larger squared error → M3 is better
        dm_stat, p_val, d_bar = dm_test_hlb(e_cml, e_m3, h=1)

        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = '(n.s.)'

        if dm_stat > 0:
            direction = 'M3 significantly better' if p_val < 0.05 else 'M3 marginally better'
        else:
            direction = 'Classical ML better' if p_val < 0.05 else 'No significant difference'

        print(f'\n  --- Results on {n_paired} paired fields (bu/acre) ---')
        print(f'  y_true range : [{y_true.min():.1f}, {y_true.max():.1f}]')
        print(f'  Classical ML  R²={r2_cml:.4f}  MAE={mae_cml:.2f}  RMSE={rmse_cml:.2f}')
        print(f'  M3 (best)     R²={r2_m3:.4f}  MAE={mae_m3:.2f}  RMSE={rmse_m3:.2f}')
        print(f'\n  DM statistic : {dm_stat:.4f}')
        print(f'  p-value      : {p_val:.4f}  {sig}')
        print(f'  Mean loss diff (CML²−M3²): {d_bar:.4f}')
        print(f'  Interpretation : {direction}')

        records.append({
            'crop': crop,
            'n_paired_fields': n_paired,
            'classical_model': 'XGBoost',
            'classical_week': cfg['week'],
            'classical_r2': round(r2_cml, 4),
            'classical_mae': round(mae_cml, 4),
            'classical_rmse': round(rmse_cml, 4),
            'm3_config': m3_modal,
            'm3_r2': round(r2_m3, 4),
            'm3_mae': round(mae_m3, 4),
            'm3_rmse': round(rmse_m3, 4),
            'dm_statistic': round(dm_stat, 4) if not np.isnan(dm_stat) else np.nan,
            'p_value': round(p_val, 4) if not np.isnan(p_val) else np.nan,
            'significance': sig,
            'mean_loss_differential': round(d_bar, 6),
            'interpretation': direction,
        })

    # --- Save results ---
    if records:
        df_out = pd.DataFrame(records)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f'\n{"="*60}')
        print(f'Results saved to {OUTPUT_CSV}')
        print(df_out.to_string(index=False))
    else:
        print('\nNo results to save.')


if __name__ == '__main__':
    run_dm_tests()
