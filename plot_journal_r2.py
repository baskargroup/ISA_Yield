import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
import os
from pathlib import Path

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

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics."""
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, p_value = pearsonr(y_true, y_pred)
    
    # Calculate bias (Mean Error)
    bias = np.mean(y_pred - y_true)
    
    return {
        'R2': r2,
        'MAPE': mape,
        'MAE': mae,
        'Pearson_r': pearson_r,
        'p_value': p_value,
        'Bias': bias,
        'N': len(y_true)
    }

def create_r2_plot(csv_path, output_dir='plots'):
    """Create a journal-quality R² plot for a single CSV file."""
    
    # Read data
    df = pd.read_csv(csv_path)
    y_true = df['YieldGT'].values
    y_pred = df['Prediction'].values
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Create figure with journal dimensions
    # Single column width: 3.5 inches, aspect ratio ~1:1
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    # Create scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=40, edgecolors='black', 
               linewidths=0.5, color='#2E86AB', zorder=3)
    
    # Calculate 1:1 line range
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    plot_min = min_val - margin
    plot_max = max_val + margin
    
    # Plot 1:1 line
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 
            'k--', linewidth=1.5, alpha=0.7, label='1:1 line', zorder=2)
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), 'r-', linewidth=1.5, alpha=0.7, 
            label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}', zorder=2)
    
    # Set axis limits
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    
    # Labels
    ax.set_xlabel('Observed Yield (bu/acre)')
    ax.set_ylabel('Predicted Yield (bu/acre)')
    
    # Get filename for title
    filename = Path(csv_path).stem
    modal_code = 'S12'
    if 'CDL' in filename:
        modal_code += 'c'
    if 'DEM' in filename:
        modal_code += 'd'
    if 'WEATHER' in filename:
        modal_code += 'w'
    if 'SOIL' in filename:
        modal_code += 's'
    crop = filename.split('_')[-1]
    tp= filename.split('_')[-2]
    # model_type = '_'.join(filename.split('_')[:-2])
    
    # ax.set_title(f'{crop} - {model_type}', fontweight='bold', pad=10)
    
    # Add metrics text box
    textstr = '\n'.join([
        f'$R^2$ = {metrics["R2"]:.3f}',
        f'MAPE = {metrics["MAPE"]:.2f}%',
        f'MAE = {metrics["MAE"]:.2f}',
        # f'Bias = {metrics["Bias"]:.2f}',
        f'n = {metrics["N"]}'
    ])
    
    # Position text box in upper left
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
    
    # Add legend
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='black', fontsize=8)
    
    # Ensure square aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{modal_code}_{crop}_{tp}_r2_plot.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'Saved: {output_path}')
    
    return metrics, output_path

def create_combined_plot(csv_files, output_dir='plots'):
    """Create a combined 2x2 plot for all CSV files (journal double-column)."""
    
    # Double column width: 7 inches
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes = axes.flatten()
    
    all_metrics = {}
    
    for idx, csv_path in enumerate(csv_files):
        if idx >= 4:  # Only plot first 4
            break
            
        ax = axes[idx]
        
        # Read data
        df = pd.read_csv(csv_path)
        y_true = df['YieldGT'].values
        y_pred = df['Prediction'].values
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Get filename for title
        filename = Path(csv_path).stem
        crop = filename.split('_')[-1]
        tp= filename.split('_')[-2]
        # model_type = '_'.join(filename.split('_')[:-1])

        all_metrics[filename] = metrics
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='black', 
                   linewidths=0.5, color='#2E86AB', zorder=3)
        
        # Calculate 1:1 line range
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05
        plot_min = min_val - margin
        plot_max = max_val + margin
        
        # Plot 1:1 line
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 
                'k--', linewidth=1.5, alpha=0.7, zorder=2)
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), 'r-', linewidth=1.5, alpha=0.7, zorder=2)
        
        # Set axis limits
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        
        # Labels
        ax.set_xlabel('Observed Yield (bu/acre)', fontsize=10)
        ax.set_ylabel('Predicted Yield (bu/acre)', fontsize=10)
        
        # Title with subplot letter
        subplot_letter = chr(97 + idx)  # a, b, c, d
        ax.set_title(f'({subplot_letter}) {crop}', fontweight='bold', pad=8, fontsize=11)
        
        # Add metrics text box
        textstr = '\n'.join([
            f'$R^2$ = {metrics["R2"]:.3f}',
            f'MAPE = {metrics["MAPE"]:.2f}%',
            f'n = {metrics["N"]}'
        ])
        
        # Position text box in upper left
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='black', linewidth=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
        
        # Ensure square aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'combined_r2_plots_{tp}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'Saved combined plot: {output_path}')
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_path = os.path.join(output_dir, 'r2_plot_metrics.csv')
    metrics_df.to_csv(metrics_path)
    print(f'Saved metrics: {metrics_path}')
    
    return all_metrics

def create_metrics_bar_plots(all_metrics, output_dir='plots'):
    """Create bar plots for R² and MAE across all models, separated by crop."""
    
    # Separate data by crop
    corn_data = {}
    soybean_data = {}
    
    for model_name, metrics in all_metrics.items():
        if 'Corn' in model_name:
            corn_data[model_name] = metrics
        else:
            soybean_data[model_name] = metrics
    
    # Process each crop separately
    for crop_name, crop_data in [('Corn', corn_data), ('Soybean', soybean_data)]:
        if not crop_data:
            continue
            
        model_names = []
        r2_values = []
        mae_values = []
        
        for model_name, metrics in crop_data.items():
            model_names.append(model_name)
            r2_values.append(metrics['R2'])
            mae_values.append(metrics['MAE'])
        
        # Create shorter, more readable labels
        short_labels = []
        for name in model_names:
            # Extract modal code
            modal_code = 'S12'
            if 'CDL' in name:
                modal_code += 'c'
            if 'DEM' in name:
                modal_code += 'd'
            if 'WEATHER' in name:
                modal_code += 'w'
            if 'SOIL' in name:
                modal_code += 's'
            short_labels.append(modal_code)
        
        # Sort by R² (descending)
        sorted_indices = np.argsort(r2_values)[::-1]
        short_labels_r2 = [short_labels[i] for i in sorted_indices]
        r2_values_sorted = [r2_values[i] for i in sorted_indices]
        
        # Color scheme
        color = '#E69F00' if crop_name == 'Corn' else '#009E73'
        
        # Create R² bar plot
        fig, ax = plt.subplots(figsize=(7, 4))
        
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, r2_values_sorted, color=color, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, r2_values_sorted)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Model Configuration', fontsize=11)
        ax.set_ylabel('R² Score', fontsize=11)
        ax.set_title(f'R² Performance Comparison - {crop_name}', fontweight='bold', fontsize=12, pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels_r2, fontsize=9)
        ax.set_ylim(0, max(r2_values_sorted) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'r2_comparison_bar_{crop_name.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'Saved R² comparison plot for {crop_name}: {output_path}')
        
        # Sort by MAE (ascending)
        sorted_indices_mae = np.argsort(mae_values)
        short_labels_mae = [short_labels[i] for i in sorted_indices_mae]
        mae_values_sorted = [mae_values[i] for i in sorted_indices_mae]
        
        # Create MAE bar plot
        fig, ax = plt.subplots(figsize=(7, 4))
        
        bars = ax.bar(x_pos, mae_values_sorted, color=color, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, mae_values_sorted)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(mae_values_sorted) * 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Model Configuration', fontsize=11)
        ax.set_ylabel('MAE (bu/acre)', fontsize=11)
        ax.set_title(f'MAE Performance Comparison - {crop_name}', fontweight='bold', fontsize=12, pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels_mae, fontsize=9)
        ax.set_ylim(0, max(mae_values_sorted) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'mae_comparison_bar_{crop_name.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'Saved MAE comparison plot for {crop_name}: {output_path}')

def create_overall_metrics_bar_plots(csv_files, output_dir='plots'):
    """Create bar plots for R² and MAE across all models, combining all data regardless of crop."""
    
    model_metrics = {}
    
    # Process each CSV file and calculate overall metrics per model
    for csv_file in csv_files:
        # Read data
        df = pd.read_csv(csv_file)
        y_true = df['YieldGT'].values
        y_pred = df['Prediction'].values
        
        # Extract model configuration from filename
        filename = Path(csv_file).stem
        
        # Extract modal code
        modal_code = 'S12'
        if 'CDL' in filename:
            modal_code += 'c'
        if 'DEM' in filename:
            modal_code += 'd'
        if 'WEATHER' in filename:
            modal_code += 'w'
        if 'SOIL' in filename:
            modal_code += 's'
        
        # Store data for this model
        if modal_code not in model_metrics:
            model_metrics[modal_code] = {'y_true': [], 'y_pred': []}
        
        model_metrics[modal_code]['y_true'].extend(y_true)
        model_metrics[modal_code]['y_pred'].extend(y_pred)
    
    # Calculate metrics for each model across all crops
    model_names = []
    r2_values = []
    mae_values = []
    
    for model_code in sorted(model_metrics.keys()):
        y_true = np.array(model_metrics[model_code]['y_true'])
        y_pred = np.array(model_metrics[model_code]['y_pred'])
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        model_names.append(model_code)
        r2_values.append(r2)
        mae_values.append(mae)
    
    # Sort by R² (descending)
    sorted_indices = np.argsort(r2_values)[::-1]
    model_names_r2 = [model_names[i] for i in sorted_indices]
    r2_values_sorted = [r2_values[i] for i in sorted_indices]
    
    # Create R² bar plot
    fig, ax = plt.subplots(figsize=(7, 4))
    
    x_pos = np.arange(len(model_names_r2))
    color = '#56B4E9'  # Light blue for overall
    bars = ax.bar(x_pos, r2_values_sorted, color=color, edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, r2_values_sorted)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model Configuration', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('R² Performance Comparison - Overall (All Crops)', fontweight='bold', fontsize=12, pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names_r2, fontsize=9)
    ax.set_ylim(0, max(r2_values_sorted) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'r2_comparison_bar_overall.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved overall R² comparison plot: {output_path}')
    
    # Sort by MAE (ascending)
    sorted_indices_mae = np.argsort(mae_values)
    model_names_mae = [model_names[i] for i in sorted_indices_mae]
    mae_values_sorted = [mae_values[i] for i in sorted_indices_mae]
    
    # Create MAE bar plot
    fig, ax = plt.subplots(figsize=(7, 4))
    
    x_pos_mae = np.arange(len(model_names_mae))
    bars = ax.bar(x_pos_mae, mae_values_sorted, color=color, edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mae_values_sorted)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(mae_values_sorted) * 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model Configuration', fontsize=11)
    ax.set_ylabel('MAE (bu/acre)', fontsize=11)
    ax.set_title('MAE Performance Comparison - Overall (All Crops)', fontweight='bold', fontsize=12, pad=15)
    ax.set_xticks(x_pos_mae)
    ax.set_xticklabels(model_names_mae, fontsize=9)
    ax.set_ylim(0, max(mae_values_sorted) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mae_comparison_bar_overall.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved overall MAE comparison plot: {output_path}')
    
    # Save overall metrics to CSV
    overall_metrics_df = pd.DataFrame({
        'Model': model_names,
        'R2': r2_values,
        'MAE': mae_values
    })
    metrics_path = os.path.join(output_dir, 'overall_metrics.csv')
    overall_metrics_df.to_csv(metrics_path, index=False)
    print(f'Saved overall metrics: {metrics_path}')

def main():
    # Define predictions directory
    predictions_dir = 'predictions_crop'
    
    # Find all CSV files
    csv_files = sorted(Path(predictions_dir).glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {predictions_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    print("-" * 60)
    
    # Create individual plots
    all_metrics = {}
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        metrics, output_path = create_r2_plot(str(csv_file))
        all_metrics[csv_file.stem] = metrics
        
        # Print metrics
        print(f"  R² = {metrics['R2']:.4f}")
        print(f"  MAPE = {metrics['MAPE']:.2f}%")
        print(f"  MAE = {metrics['MAE']:.2f}")
        print(f"  Bias = {metrics['Bias']:.2f}")
        print(f"  N = {metrics['N']}")
    
    print("\n" + "=" * 60)
    print("Creating combined plot...")
    create_combined_plot(csv_files)
    
    print("\n" + "=" * 60)
    print("Creating metrics comparison bar plots...")
    create_metrics_bar_plots(all_metrics)
    
    print("\n" + "=" * 60)
    print("Creating overall metrics bar plots (all crops combined)...")
    create_overall_metrics_bar_plots(csv_files)
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print(f"Output directory: plots/")

if __name__ == '__main__':
    main()
