#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prediction Analysis Script
Exported from prediction_analysis.ipynb
"""

import os
import glob
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Set journal-standard matplotlib parameters
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.titlesize'] = 11
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['patch.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['xtick.major.size'] = 3.5
mpl.rcParams['ytick.major.size'] = 3.5
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05

# Journal column widths (in inches)
SINGLE_COLUMN_WIDTH = 3.5  # ~89mm
DOUBLE_COLUMN_WIDTH = 7.0  # ~178mm

# ============================================================================
# Section 1: Process Prediction Files
# ============================================================================

# Directory containing prediction CSV files
predictions_dir = 'predictions_final/'

# Get all CSV files in the predictions directory
csv_files = glob.glob(os.path.join(predictions_dir, '*_DEM_*.csv'))

# Dictionary to store results
results = []

# Process each CSV file
for csv_file in csv_files:
    # Extract filename without path and extension
    base_filename = os.path.basename(csv_file)
    file_key = base_filename.replace('.csv', '')
    
    # Extract time from filename (last part after last underscore)
    try:
        time_value = int(file_key.split('_')[-1])
    except (ValueError, IndexError):
        print(f"Skipping {base_filename}: Cannot extract time value")
        continue
    
    print(f"Processing: {base_filename} (time={time_value})")
    
    # Read and process data
    df = pd.read_csv(csv_file)
    df = df[df['YieldGT'] > 0]
    
    # Aggregate by filename
    df_agg = df.groupby('Filename').agg({
        'YieldGT': 'mean',
        'Prediction': 'mean'
    }).reset_index()
    
    # Calculate metrics
    if len(df_agg) > 0:
        mae = mean_absolute_error(df_agg['YieldGT'], df_agg['Prediction'])
        mape = np.mean(np.abs((df_agg['YieldGT'] - df_agg['Prediction']) / df_agg['YieldGT'])) * 100
        r2 = r2_score(df_agg['YieldGT'], df_agg['Prediction'])
        
        results.append({
            'filename': file_key,
            'time': time_value,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'n_samples': len(df_agg)
        })
        
        print(f"  R²: {r2:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, n={len(df_agg)}")
    else:
        print(f"  Skipped: No valid data")

# Convert results to DataFrame
results_df_wd = pd.DataFrame(results)
results_df_wd = results_df_wd.sort_values('time')
results_df_wd.reset_index(drop=True, inplace=True)
results_df_wd['mae'] = results_df_wd['mae'] * 370  # Convert to Bu/Acre

print(f"\nProcessed {len(results_df_wd)} files successfully")
print(results_df_wd.head(15))

# ============================================================================
# Section 2: Plot Classical ML Results
# ============================================================================

_code = "s12wd"  # Specify the code used in filenames
# Load results
df = pd.read_csv(f"classical_ml_results_{_code}.csv")
df["biweek_num"] = df["biweek"].str.extract(r"(\d+)").astype(int)
df['test_mae'] = df['test_mae'] * 370

# Prepare R² data
r2_long = df.melt(
    id_vars=["biweek_num", "model"],
    value_vars=["test_r2"],
    var_name="split",
    value_name="R²"
)
r2_long["split"] = r2_long["split"].str.replace("_r2", "").str.capitalize()

# Plot R²
plt.figure(figsize=(3.5, 3.5))
sns.lineplot(
    data=r2_long,
    x="biweek_num",
    y="R²",
    hue="model",
    marker="o",
    palette="Set1"
)
plt.title("Model $R^2$ Across Biweeks", fontsize=10, pad=10)
plt.ylim(0.0, 1.0)
plt.xlabel("Biweek", fontsize=9)
plt.ylabel("$R^2$ Score", fontsize=9)
plt.xticks(df["biweek_num"].unique(), fontsize=8)
plt.yticks(fontsize=8)
plt.legend(title="Model", loc="best", fontsize=7, title_fontsize=8)
plt.tight_layout()
plt.savefig("classical_ml_r2_biweek.pdf", bbox_inches="tight")
plt.show()

# Prepare MAE data
mae_long = df.melt(
    id_vars=["biweek_num", "model"],
    value_vars=["test_mae"],
    var_name="split",
    value_name="MAE"
)
mae_long["split"] = mae_long["split"].str.replace("_mae", "").str.capitalize()

# Plot MAE
plt.figure(figsize=(3.5, 3.5))
sns.lineplot(
    data=mae_long,
    x="biweek_num",
    y="MAE",
    hue="model",
    marker="o",
    palette="Set1"
)
plt.title("Model MAE Across Biweeks", fontsize=10, pad=10)
plt.xlabel("Biweek", fontsize=9)
plt.ylabel("MAE (Bu/Acre)", fontsize=9)
plt.xticks(df["biweek_num"].unique(), fontsize=8)
plt.yticks(fontsize=8)
plt.ylim(20, 65)
plt.legend(title="Model", loc="best", fontsize=7, title_fontsize=8)
plt.tight_layout()
plt.savefig("classical_ml_mae_biweek.pdf", bbox_inches="tight")
plt.show()

# ============================================================================
# Section 3: Compare TerraMind vs Classical ML
# ============================================================================

# Load classical ML results for comparison
_code = "s12wd"  # Specify the code used in filenames
df_classical = pd.read_csv(f"classical_ml_results_{_code}.csv")
df_classical["biweek_num"] = df_classical["biweek"].str.extract(r"(\d+)").astype(int)
df_classical['test_mae'] = df_classical['test_mae'] * 370  # Convert to Bu/Acre

# Get best model results from classical ML (e.g., XgBoost)
df_classical_rf = df_classical[df_classical['model'] == 'XGBoost'].reset_index(drop=True)
df_classical_rf = df_classical_rf.sort_values('biweek_num').reset_index(drop=True)
df_classical_plsr = df_classical[df_classical['model'] == 'PLSR'].reset_index(drop=True)
df_classical_plsr = df_classical_plsr.sort_values('biweek_num').reset_index(drop=True)

# Plot 1: R² comparison
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

ax.plot(results_df_wd['time'], results_df_wd['r2'], marker='o', linewidth=1.5, 
        markersize=5, color='#2E86AB', label='TerraMind', alpha=0.8)
ax.plot(df_classical_rf['biweek_num'], df_classical_rf['test_r2'], 
        marker='s', linewidth=1.5, markersize=5, color='#A23B72', 
        label='XgBoost', alpha=0.8)
ax.plot(df_classical_plsr['biweek_num'], df_classical_plsr['test_r2'], 
        marker='^', linewidth=1.5, markersize=5, color='#D95F02', 
        label='PLSR', alpha=0.8)

ax.set_xlabel('Biweek', fontsize=9)
ax.set_ylabel('$R^2$ Score', fontsize=9)
# ax.set_title('$R^2$ Score Comparison', fontsize=10, pad=10)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.3)
ax.set_ylim([0, 1.05])
ax.legend(loc='best', frameon=True, fontsize=7, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('figures/dlS12WD_vs_classical_r2.png', bbox_inches='tight')
plt.savefig('figures/dlS12WD_vs_classical_r2.pdf', bbox_inches='tight')
plt.show()

# Plot 2: MAE comparison
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

ax.plot(results_df_wd['time'], results_df_wd['mae'], marker='o', linewidth=1.5, 
        markersize=5, color='#2E86AB', label='TerraMind', alpha=0.8)
ax.plot(df_classical_rf['biweek_num'], df_classical_rf['test_mae'], 
        marker='s', linewidth=1.5, markersize=5, color='#A23B72', 
        label='XgBoost', alpha=0.8)
ax.plot(df_classical_plsr['biweek_num'], df_classical_plsr['test_mae'], 
        marker='^', linewidth=1.5, markersize=5, color='#D95F02', 
        label='PLSR', alpha=0.8)

ax.set_xlabel('Biweek', fontsize=9)
ax.set_ylabel('MAE (Bu/Acre)', fontsize=9)
# ax.set_title('MAE Comparison', fontsize=10, pad=10)
ax.legend(loc='best', frameon=True, fontsize=7, edgecolor='gray')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('figures/dlS12WD_vs_classical_mae.png', bbox_inches='tight')
plt.savefig('figures/dlS12WD_vs_classical_mae.pdf', bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("TerraMind SUMMARY:")
print("="*60)
print(f"Mean R²: {results_df_wd['r2'].mean():.4f} (±{results_df_wd['r2'].std():.4f})")
print(f"Mean MAE: {results_df_wd['mae'].mean():.4f} (±{results_df_wd['mae'].std():.4f})")
print(f"Best R²: {results_df_wd['r2'].max():.4f} at biweek {results_df_wd.loc[results_df_wd['r2'].idxmax(), 'time']}")
print(f"Best MAE: {results_df_wd['mae'].min():.4f} at biweek {results_df_wd.loc[results_df_wd['mae'].idxmin(), 'time']}")

print("\n" + "="*60)
print("XGBoost SUMMARY:")
print("="*60)
print(f"Mean R²: {df_classical_rf['test_r2'].mean():.4f} (±{df_classical_rf['test_r2'].std():.4f})")
print(f"Mean MAE: {df_classical_rf['test_mae'].mean():.4f} (±{df_classical_rf['test_mae'].std():.4f})")
print(f"Best R²: {df_classical_rf['test_r2'].max():.4f} at biweek {df_classical_rf.loc[df_classical_rf['test_r2'].idxmax(), 'biweek_num']}")
print(f"Best MAE: {df_classical_rf['test_mae'].min():.4f} at biweek {df_classical_rf.loc[df_classical_rf['test_mae'].idxmin(), 'biweek_num']}")
print("="*60)

print("\n" + "="*60)
print("PLSR SUMMARY:")
print("="*60)
print(f"Mean R²: {df_classical_plsr['test_r2'].mean():.4f} (±{df_classical_plsr['test_r2'].std():.4f})")
print(f"Mean MAE: {df_classical_plsr['test_mae'].mean():.4f} (±{df_classical_plsr['test_mae'].std():.4f})")
print(f"Best R²: {df_classical_plsr['test_r2'].max():.4f} at biweek {df_classical_plsr.loc[df_classical_plsr['test_r2'].idxmax(), 'biweek_num']}")
print(f"Best MAE: {df_classical_plsr['test_mae'].min():.4f} at biweek {df_classical_plsr.loc[df_classical_plsr['test_mae'].idxmin(), 'biweek_num']}")
print("="*60)
