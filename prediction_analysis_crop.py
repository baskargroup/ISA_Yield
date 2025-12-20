import os
import glob
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pdb

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

# Directory containing prediction CSV files
predictions_dir = 'predictions_crop/'

# Get all CSV files in the predictions directory
csv_files = glob.glob(os.path.join(predictions_dir, '*_DEM_*.csv'))

# Dictionaries to store results for each crop
results_corn = []
results_soybean = []

# Process each CSV file
for csv_file in csv_files:
    # Extract filename without path and extension
    base_filename = os.path.basename(csv_file)
    file_key = base_filename.replace('.csv', '')
    # Extract time from filename (last part after last underscore)
    try:
        time_value = int(file_key.split('_')[-2])
    except (ValueError, IndexError):
        print(f"Skipping {base_filename}: Cannot extract time value")
        continue
    
    print(f"Processing: {base_filename} (time={time_value})")
    
    # Read and process data
    df = pd.read_csv(csv_file)
    df = df[df['YieldGT'] > 0]
    
    # Separate corn and soybean data based on Filename column
    df_corn = df[df['Filename'].str.contains('Corn', case=False, na=False)]
    df_soybean = df[df['Filename'].str.contains('Soybean', case=False, na=False)]
    
    # Process Corn data
    if len(df_corn) > 0:
        df_corn_agg = df_corn.groupby('Filename').agg({
            'YieldGT': 'mean',
            'Prediction': 'mean'
        }).reset_index()
        
        if len(df_corn_agg) > 0:
            mae_corn = mean_absolute_error(df_corn_agg['YieldGT'], df_corn_agg['Prediction'])
            mape_corn = np.mean(np.abs((df_corn_agg['YieldGT'] - df_corn_agg['Prediction']) / df_corn_agg['YieldGT'])) * 100
            r2_corn = r2_score(df_corn_agg['YieldGT'], df_corn_agg['Prediction'])
            
            results_corn.append({
                'filename': file_key,
                'time': time_value,
                'r2': r2_corn,
                'mae': mae_corn,
                'mape': mape_corn,
                'n_samples': len(df_corn_agg)
            })
            
            print(f"  Corn - R²: {r2_corn:.4f}, MAE: {mae_corn:.4f}, MAPE: {mape_corn:.2f}%, n={len(df_corn_agg)}")
    
    # Process Soybean data
    if len(df_soybean) > 0:
        df_soybean_agg = df_soybean.groupby('Filename').agg({
            'YieldGT': 'mean',
            'Prediction': 'mean'
        }).reset_index()
        
        if len(df_soybean_agg) > 0:
            mae_soybean = mean_absolute_error(df_soybean_agg['YieldGT'], df_soybean_agg['Prediction'])
            mape_soybean = np.mean(np.abs((df_soybean_agg['YieldGT'] - df_soybean_agg['Prediction']) / df_soybean_agg['YieldGT'])) * 100
            r2_soybean = r2_score(df_soybean_agg['YieldGT'], df_soybean_agg['Prediction'])
            
            results_soybean.append({
                'filename': file_key,
                'time': time_value,
                'r2': r2_soybean,
                'mae': mae_soybean,
                'mape': mape_soybean,
                'n_samples': len(df_soybean_agg)
            })
            
            print(f"  Soybean - R²: {r2_soybean:.4f}, MAE: {mae_soybean:.4f}, MAPE: {mape_soybean:.2f}%, n={len(df_soybean_agg)}")

# Convert results to DataFrames
results_df_corn = pd.DataFrame(results_corn)
results_df_corn = results_df_corn.sort_values('time').reset_index(drop=True)

results_df_soybean = pd.DataFrame(results_soybean)
results_df_soybean = results_df_soybean.sort_values('time').reset_index(drop=True)

print(f"\nProcessed {len(results_df_corn)} corn files and {len(results_df_soybean)} soybean files successfully")
print("\nCorn Results:")
print(results_df_corn.head(15))
print("\nSoybean Results:")
print(results_df_soybean.head(15))

# ============= R² PLOTS =============
# Create figure for R² plots (Corn and Soybean side by side)
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))

# Plot 1: Corn R² over time
ax1.plot(results_df_corn['time'], results_df_corn['r2'], marker='o', linewidth=2.5, 
         markersize=8, color='#F4A460', label='Corn', alpha=0.8)
ax1.set_xlabel('Biweek', fontsize=11)
ax1.set_ylabel('R² Score', fontsize=11)
ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
ax1.set_ylim([0, 1.05])
ax1.legend(loc='best', frameon=True, fontsize=9, edgecolor='gray')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 2: Soybean R² over time
ax2.plot(results_df_soybean['time'], results_df_soybean['r2'], marker='s', linewidth=2.5, 
         markersize=8, color='#228B22', label='Soybean', alpha=0.8)
ax2.set_xlabel('Biweek', fontsize=11)
ax2.set_ylabel('R² Score', fontsize=11)
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
ax2.set_ylim([0, 1.05])
ax2.legend(loc='best', frameon=True, fontsize=9, edgecolor='gray')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save R² figure
plt.savefig('figures/crop_r2_comparison.png', bbox_inches='tight')
plt.savefig('figures/crop_r2_comparison.pdf', bbox_inches='tight')
plt.show()

# ============= MAE PLOTS =============
# Create figure for MAE plots (Corn and Soybean side by side)
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
# Plot 3: Corn MAE over time
ax3.plot(results_df_corn['time'], results_df_corn['mae'], marker='o', linewidth=2.5, 
         markersize=8, color='#F4A460', label='Corn', alpha=0.8)
ax3.set_xlabel('Biweek', fontsize=11)
ax3.set_ylabel('MAE (Bu/Acre)', fontsize=11)
ax3.legend(loc='best', frameon=True, fontsize=9, edgecolor='gray')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 4: Soybean MAE over time
ax4.plot(results_df_soybean['time'], results_df_soybean['mae'], marker='s', linewidth=2.5, 
         markersize=8, color='#228B22', label='Soybean', alpha=0.8)
ax4.set_xlabel('Biweek', fontsize=11)
ax4.set_ylabel('MAE (Bu/Acre)', fontsize=11)
ax4.legend(loc='best', frameon=True, fontsize=9, edgecolor='gray')
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save MAE figure
plt.savefig('figures/crop_mae_comparison.png', bbox_inches='tight')
plt.savefig('figures/crop_mae_comparison.pdf', bbox_inches='tight')
plt.show()

# ============= COMBINED COMPARISON PLOTS =============
# Create separate figures comparing corn and soybean overlaid

# Figure 3: R² comparison between Corn and Soybean (overlaid)
fig3, ax5 = plt.subplots(1, 1, figsize=(3.5, 3.5))
ax5.plot(results_df_corn['time'], results_df_corn['r2'], marker='o', linewidth=2.5, 
         markersize=8, color='#F4A460', label='Corn', alpha=0.8)
ax5.plot(results_df_soybean['time'], results_df_soybean['r2'], marker='s', linewidth=2.5, 
         markersize=8, color='#228B22', label='Soybean', alpha=0.8)
ax5.set_xlabel('Biweek', fontsize=11)
ax5.set_ylabel('R² Score', fontsize=11)
ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
ax5.set_ylim([0, 1.05])
ax5.legend(loc='best', frameon=True, fontsize=9, edgecolor='gray')
ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save R² comparison figure
plt.savefig('figures/combined_r2_comparison.png', bbox_inches='tight')
plt.savefig('figures/combined_r2_comparison.pdf', bbox_inches='tight')
plt.show()

# Figure 4: MAE comparison between Corn and Soybean (overlaid)
fig4, ax6 = plt.subplots(1, 1, figsize=(3.5, 3.5))
ax6.plot(results_df_corn['time'], results_df_corn['mae'], marker='o', linewidth=2.5, 
         markersize=8, color='#F4A460', label='Corn', alpha=0.8)
ax6.plot(results_df_soybean['time'], results_df_soybean['mae'], marker='s', linewidth=2.5, 
         markersize=8, color='#228B22', label='Soybean', alpha=0.8)
ax6.set_xlabel('Biweek', fontsize=11)
ax6.set_ylabel('MAE (Bu/Acre)', fontsize=11)
ax6.legend(loc='best', frameon=True, fontsize=9, edgecolor='gray')
ax6.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save MAE comparison figure
plt.savefig('figures/combined_mae_comparison.png', bbox_inches='tight')
plt.savefig('figures/combined_mae_comparison.pdf', bbox_inches='tight')
plt.show()

