import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
corn_df = pd.read_csv('Terramind_corn.csv')
soybean_df = pd.read_csv('Terramind_soybean.csv')

# Extract experiment names (remove spaces and make comparable)
corn_df['Experiment'] = corn_df['Name'].str.replace('S12WD', 'S12wd').str.replace('_', ' ')
soybean_df['Experiment'] = soybean_df['Name'].str.replace('_', ' ')

# Split Name column and use the 2nd element (index 1) for sorting
corn_df['sort_key'] = corn_df['Name'].str.split('_').str[1].astype(int)
soybean_df['sort_key'] = soybean_df['Name'].str.split('_').str[1].astype(int)

# Sort by the second element
corn_df = corn_df.sort_values('sort_key')
soybean_df = soybean_df.sort_values('sort_key')

# Denormalize MAE values
# Based on the normalization: normalized = (data - data_min) / (data_max - data_min)
# To denormalize: data = normalized * (data_max - data_min) + data_min
norm_max_corn = 370.0  # Typical max for corn yield
norm_max_soybean = 150.0  # Typical max for soybean yield
data_min_corn = 50.0
data_min_soybean = 30.0

corn_df['val/MAE (Min)'] = corn_df['val/MAE (Min)'] * (norm_max_corn - data_min_corn) + data_min_corn
soybean_df['val/MAE (Min)'] = soybean_df['val/MAE (Min)'] * (norm_max_soybean - data_min_soybean) + data_min_soybean

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: val/MAE (Min)
ax1 = axes[0]
x = np.arange(len(corn_df))

ax1.plot(x, corn_df['val/MAE (Min)'], marker='o', linewidth=2.5, markersize=8, 
         label='Corn', alpha=0.8, color='#2E86AB')
ax1.plot(x, soybean_df['val/MAE (Min)'], marker='s', linewidth=2.5, markersize=8, 
         label='Soybean', alpha=0.8, color='#A23B72')

ax1.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax1.set_ylabel('val/MAE (Min) - Bu/Acre', fontsize=12, fontweight='bold')
ax1.set_title('Validation MAE Comparison: Corn vs Soybean (Denormalized)', fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(corn_df['Experiment'], rotation=45, ha='right')
ax1.legend(fontsize=11, loc='best', frameon=True, edgecolor='gray')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: val/R2_Score (Max)
ax2 = axes[1]
ax2.plot(x, corn_df['val/R2_Score (Max)'], marker='o', linewidth=2.5, markersize=8, 
         label='Corn', alpha=0.8, color='#2E86AB')
ax2.plot(x, soybean_df['val/R2_Score (Max)'], marker='s', linewidth=2.5, markersize=8, 
         label='Soybean', alpha=0.8, color='#A23B72')

ax2.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax2.set_ylabel('val/R2_Score (Max)', fontsize=12, fontweight='bold')
ax2.set_title('Validation R² Score Comparison: Corn vs Soybean', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(corn_df['Experiment'], rotation=45, ha='right')
ax2.legend(fontsize=11, loc='best', frameon=True, edgecolor='gray')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_ylim(0, 1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('corn_vs_soybean_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('corn_vs_soybean_comparison.pdf', bbox_inches='tight')
print("Plots saved as 'corn_vs_soybean_comparison.png' and 'corn_vs_soybean_comparison.pdf'")

# Print summary statistics
print("\n=== Summary Statistics ===")
print("\nval/MAE (Min):")
print(f"Corn    - Mean: {corn_df['val/MAE (Min)'].mean():.4f}, Std: {corn_df['val/MAE (Min)'].std():.4f}, Min: {corn_df['val/MAE (Min)'].min():.4f}, Max: {corn_df['val/MAE (Min)'].max():.4f}")
print(f"Soybean - Mean: {soybean_df['val/MAE (Min)'].mean():.4f}, Std: {soybean_df['val/MAE (Min)'].std():.4f}, Min: {soybean_df['val/MAE (Min)'].min():.4f}, Max: {soybean_df['val/MAE (Min)'].max():.4f}")

print("\nval/R2_Score (Max):")
print(f"Corn    - Mean: {corn_df['val/R2_Score (Max)'].mean():.4f}, Std: {corn_df['val/R2_Score (Max)'].std():.4f}, Min: {corn_df['val/R2_Score (Max)'].min():.4f}, Max: {corn_df['val/R2_Score (Max)'].max():.4f}")
print(f"Soybean - Mean: {soybean_df['val/R2_Score (Max)'].mean():.4f}, Std: {soybean_df['val/R2_Score (Max)'].std():.4f}, Min: {soybean_df['val/R2_Score (Max)'].min():.4f}, Max: {soybean_df['val/R2_Score (Max)'].max():.4f}")

plt.show()
