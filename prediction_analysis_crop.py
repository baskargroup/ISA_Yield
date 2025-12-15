import os
import glob
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing prediction CSV files
predictions_dir = 'predictions/'

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
        time_value = int(file_key.split('_')[-1])
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

# Denormalize MAE using min-max scaling
min_corn = 50
max_corn = 370
min_soybean = 30
max_soybean = 150

results_df_corn['mae'] = results_df_corn['mae'] * (max_corn - min_corn) + min_corn
results_df_soybean['mae'] = results_df_soybean['mae'] * (max_soybean - min_soybean) + min_soybean

print(f"\nProcessed {len(results_df_corn)} corn files and {len(results_df_soybean)} soybean files successfully")
print("\nCorn Results:")
print(results_df_corn.head(15))
print("\nSoybean Results:")
print(results_df_soybean.head(15))


# Set style for publication
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Create figure with 4 subplots (2x2 grid)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# ============= CORN PLOTS =============
# Plot 1: Corn R² over time
ax1.plot(results_df_corn['time'], results_df_corn['r2'], marker='o', linewidth=2.5, 
         markersize=8, color='#F4A460', label='Corn', alpha=0.8)
ax1.set_xlabel('Biweek', fontsize=14, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax1.set_title('Corn: R² Score Over Time', fontsize=14, fontweight='bold', pad=15)
ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
ax1.set_ylim([0, 1.05])
ax1.legend(loc='best', frameon=True, fontsize=11, edgecolor='gray')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 2: Corn MAE over time
ax2.plot(results_df_corn['time'], results_df_corn['mae'], marker='o', linewidth=2.5, 
         markersize=8, color='#F4A460', label='Corn', alpha=0.8)
ax2.set_xlabel('Biweek', fontsize=14, fontweight='bold')
ax2.set_ylabel('MAE (Bu/Acre)', fontsize=14, fontweight='bold')
ax2.set_title('Corn: MAE Over Time', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='best', frameon=True, fontsize=11, edgecolor='gray')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# ============= SOYBEAN PLOTS =============
# Plot 3: Soybean R² over time
ax3.plot(results_df_soybean['time'], results_df_soybean['r2'], marker='s', linewidth=2.5, 
         markersize=8, color='#228B22', label='Soybean', alpha=0.8)
ax3.set_xlabel('Biweek', fontsize=14, fontweight='bold')
ax3.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax3.set_title('Soybean: R² Score Over Time', fontsize=14, fontweight='bold', pad=15)
ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
ax3.set_ylim([0, 1.05])
ax3.legend(loc='best', frameon=True, fontsize=11, edgecolor='gray')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 4: Soybean MAE over time
ax4.plot(results_df_soybean['time'], results_df_soybean['mae'], marker='s', linewidth=2.5, 
         markersize=8, color='#228B22', label='Soybean', alpha=0.8)
ax4.set_xlabel('Biweek', fontsize=14, fontweight='bold')
ax4.set_ylabel('MAE (Bu/Acre)', fontsize=14, fontweight='bold')
ax4.set_title('Soybean: MAE Over Time', fontsize=14, fontweight='bold', pad=15)
ax4.legend(loc='best', frameon=True, fontsize=11, edgecolor='gray')
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save figure
plt.savefig('figures/crop_comparison_corn_vs_soybean.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/crop_comparison_corn_vs_soybean.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ============= COMBINED COMPARISON PLOTS =============
# Create a second figure comparing corn and soybean directly
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Plot 5: R² comparison between Corn and Soybean
ax5.plot(results_df_corn['time'], results_df_corn['r2'], marker='o', linewidth=2.5, 
         markersize=8, color='#F4A460', label='Corn', alpha=0.8)
ax5.plot(results_df_soybean['time'], results_df_soybean['r2'], marker='s', linewidth=2.5, 
         markersize=8, color='#228B22', label='Soybean', alpha=0.8)
ax5.set_xlabel('Biweek', fontsize=14, fontweight='bold')
ax5.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax5.set_title('R² Score Comparison: Corn vs Soybean', fontsize=14, fontweight='bold', pad=15)
ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
ax5.set_ylim([0, 1.05])
ax5.legend(loc='best', frameon=True, fontsize=11, edgecolor='gray')
ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 6: MAE comparison between Corn and Soybean
ax6.plot(results_df_corn['time'], results_df_corn['mae'], marker='o', linewidth=2.5, 
         markersize=8, color='#F4A460', label='Corn', alpha=0.8)
ax6.plot(results_df_soybean['time'], results_df_soybean['mae'], marker='s', linewidth=2.5, 
         markersize=8, color='#228B22', label='Soybean', alpha=0.8)
ax6.set_xlabel('Biweek', fontsize=14, fontweight='bold')
ax6.set_ylabel('MAE (Bu/Acre)', fontsize=14, fontweight='bold')
ax6.set_title('MAE Comparison: Corn vs Soybean', fontsize=14, fontweight='bold', pad=15)
ax6.legend(loc='best', frameon=True, fontsize=11, edgecolor='gray')
ax6.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save figure
plt.savefig('figures/combined_crop_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/combined_crop_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

