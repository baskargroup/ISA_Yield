#!/usr/bin/env python3
"""
Generate publication-quality figures for ISA Yield Prediction paper.
Models: M1 (frozen backbone, S12d), M2 (unfrozen, S12wdsc), M3 (modality ablation),
        M4 (temporal ablation), M5 (different normalization)
Data: Weekly only (24 weeks). No biweekly. No old_results.

Output: journal_figures/ with PDF + PNG at 600 DPI
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# JOURNAL-STANDARD MATPLOTLIB RCPARAMS
# ============================================================================
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.titlesize': 11,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'figure.dpi': 150,
    'mathtext.default': 'regular',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,       # TrueType fonts in PDF
    'ps.fonttype': 42,
})

# Column widths (inches) — standard single/double column for journals
SINGLE_COL = 3.5    # ~89 mm
ONE_HALF_COL = 5.5  # ~140 mm
DOUBLE_COL = 7.2    # ~183 mm

# Color palettes
CORN_COLOR = '#E8A838'
SOYBEAN_COLOR = '#56A764'
NEUTRAL_BLUE = '#2E86AB'
ACCENT_RED = '#D64045'
ACCENT_PURPLE = '#7B68EE'
SCATTER_COLOR = '#3A7CA5'
DARK_GRAY = '#333333'

# Model-specific colors
MODEL_COLORS = {
    'M1': '#4C72B0',
    'M2': '#DD8452',
    'M3': '#55A868',
    'M4': '#C44E52',
    'M5': '#8172B3',
}

# Modality colors for M3 ablation
MODALITY_COLORS = {
    'S12': '#4C72B0',
    'S12c': '#64B5F6',
    'S12d': '#55A868',
    'S12s': '#FF7043',
    'S12w': '#AB47BC',
    'S12cd': '#26A69A',
    'S12cs': '#EF5350',
    'S12cw': '#5C6BC0',
    'S12ds': '#66BB6A',
    'S12dw': '#FFA726',
    'S12ws': '#EC407A',
    'S12cdw': '#D4E157',
    'S12cds': '#42A5F5',
    'S12cws': '#26C6DA',
    'S12dws': '#FFCA28',
    'S12cdws': '#78909C',
}

OUTPUT_DIR = 'journal_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, name, tight=True):
    """Save figure in both PDF and PNG."""
    if tight:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'), bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), bbox_inches='tight', facecolor='white')
    else:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'), facecolor='white')
        fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), facecolor='white')
    plt.close(fig)
    print(f'  Saved: {name}.pdf / .png')


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    pearson_r, p_val = pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    bias = np.mean(y_pred - y_true)
    return dict(R2=r2, MAE=mae, RMSE=rmse, MAPE=mape,
                Pearson_r=pearson_r, p_value=p_val, Bias=bias, N=len(y_true))


def extract_modal_code(filename):
    """Extract short modality code from prediction filename."""
    code = 'S12'
    if 'CDL' in filename:
        code += 'c'
    if 'DEM' in filename:
        code += 'd'
    if 'WEATHER' in filename:
        code += 'w'
    if 'SOIL' in filename:
        code += 's'
    return code


def extract_crop(filename):
    """Extract crop from prediction filename."""
    if 'Corn' in filename:
        return 'Corn'
    elif 'Soybean' in filename:
        return 'Soybean'
    return 'Unknown'


# ============================================================================
# FIGURE 1a: Corn Yield Distribution by Year
# ============================================================================
def fig1a_yield_distributions_corn():
    """Corn yield distribution histograms by year (2017-2024).
    Aggregated to field-level (mean yield per Layer_ID)."""
    print('Figure 1a: Corn yield distributions by year...')

    years = list(range(2017, 2025))
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, 3.6),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for j, yr in enumerate(years):
        ppath = f'Yield_{yr}_filtered.parquet'
        ax = axes_flat[j]
        if not os.path.exists(ppath):
            ax.set_title(str(yr), fontweight='bold', pad=3)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, color='gray')
            continue

        df = pd.read_parquet(ppath)
        field_df = df.groupby(['Layer_ID', 'Crop']).agg(
            mean_yield=('Yield', 'mean')).reset_index()
        corn = field_df[field_df['Crop'].str.contains('Corn', case=False, na=False)]['mean_yield']
        corn = corn[(corn > 10) & (corn < 400)]

        n_bins = min(25, max(5, len(corn) // 3))
        ax.hist(corn, bins=n_bins, color=CORN_COLOR, alpha=0.85, edgecolor='white',
                linewidth=0.3, density=False)
        ax.set_title(str(yr), fontweight='bold', pad=3)

        if len(corn) > 0:
            txt = f'$\\mu$={corn.mean():.0f}\n$\\sigma$={corn.std():.0f}\nn={len(corn)}'
            ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=6,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray', lw=0.4))

    for j in range(len(years), nrows * ncols):
        axes_flat[j].set_visible(False)

    for j in range(nrows):
        axes[j, 0].set_ylabel('No. of Fields')
    for j in range(ncols):
        axes[-1, j].set_xlabel('Yield (bu/acre)')

    fig.suptitle('Corn — Field-Level Yield Distributions', fontweight='bold', fontsize=9, y=1.01)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    save_fig(fig, 'fig1a_yield_distributions_corn')


# ============================================================================
# FIGURE 1b: Soybean Yield Distribution by Year
# ============================================================================
def fig1b_yield_distributions_soybean():
    """Soybean yield distribution histograms by year (2017-2024).
    Aggregated to field-level (mean yield per Layer_ID)."""
    print('Figure 1b: Soybean yield distributions by year...')

    years = list(range(2017, 2025))
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, 3.6),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for j, yr in enumerate(years):
        ppath = f'Yield_{yr}_filtered.parquet'
        ax = axes_flat[j]
        if not os.path.exists(ppath):
            ax.set_title(str(yr), fontweight='bold', pad=3)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, color='gray')
            continue

        df = pd.read_parquet(ppath)
        field_df = df.groupby(['Layer_ID', 'Crop']).agg(
            mean_yield=('Yield', 'mean')).reset_index()
        soy = field_df[field_df['Crop'].str.contains('Soy', case=False, na=False)]['mean_yield']
        soy = soy[(soy > 5) & (soy < 150)]

        n_bins = min(25, max(5, len(soy) // 3))
        ax.hist(soy, bins=n_bins, color=SOYBEAN_COLOR, alpha=0.85, edgecolor='white',
                linewidth=0.3, density=False)
        ax.set_title(str(yr), fontweight='bold', pad=3)

        if len(soy) > 0:
            txt = f'$\\mu$={soy.mean():.0f}\n$\\sigma$={soy.std():.0f}\nn={len(soy)}'
            ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=6,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray', lw=0.4))

    for j in range(len(years), nrows * ncols):
        axes_flat[j].set_visible(False)

    for j in range(nrows):
        axes[j, 0].set_ylabel('No. of Fields')
    for j in range(ncols):
        axes[-1, j].set_xlabel('Yield (bu/acre)')

    fig.suptitle('Soybean — Field-Level Yield Distributions', fontweight='bold', fontsize=9, y=1.01)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    save_fig(fig, 'fig1b_yield_distributions_soybean')


# ============================================================================
# FIGURE 2: Data Modality Overview — Band statistics
# ============================================================================
def fig2_modality_band_statistics():
    """Show band-level statistics across modalities from processed data."""
    print('Figure 2: Modality band statistics...')

    base = 'processed_data/weekly_24/processed_data_weekly_24'
    modalities = {
        'S2L2A': {'bands': 12, 'label': 'Sentinel-2',
                  'names': ['B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','SCL']},
        'S1GRD': {'bands': 2, 'label': 'Sentinel-1', 'names': ['VV','VH']},
        'DEM':   {'bands': 1, 'label': 'DEM', 'names': ['Elevation']},
        'WEATHER': {'bands': 7, 'label': 'Weather',
                    'names': ['DAYL','PRCP','SRAD','SWE','TMAX','TMIN','VP']},
        'SOIL':  {'bands': 10, 'label': 'Soil',
                  'names': [f'Soil_{i+1}' for i in range(10)]},
        'CDL':   {'bands': 1, 'label': 'CDL', 'names': ['CropType']},
    }

    # Sample a subset of files for speed
    sample_files = sorted(os.listdir(os.path.join(base, 'S2L2A')))[:30]

    stats = {}
    for mod_name, mod_info in modalities.items():
        mod_dir = os.path.join(base, mod_name)
        band_means = [[] for _ in range(mod_info['bands'])]

        for fname in sample_files:
            fpath = os.path.join(mod_dir, fname)
            if os.path.exists(fpath):
                arr = np.load(fpath)  # (24, C, H, W)
                for b in range(min(mod_info['bands'], arr.shape[1])):
                    vals = arr[:, b].flatten()
                    valid = vals[~np.isnan(vals)]
                    if len(valid) > 0:
                        band_means[b].append(np.mean(valid))

        stats[mod_name] = {
            'means': [np.mean(bm) if bm else 0 for bm in band_means],
            'stds': [np.std(bm) if bm else 0 for bm in band_means],
            'info': mod_info,
        }

    # Create multi-panel figure: one panel per modality
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 4.0))
    axes = axes.flatten()
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#CCB974']

    for idx, (mod_name, data) in enumerate(stats.items()):
        ax = axes[idx]
        names = data['info']['names']
        means = data['means']
        stds = data['stds']
        x = np.arange(len(means))

        bars = ax.bar(x, means, yerr=stds, capsize=2, color=colors[idx],
                      alpha=0.8, edgecolor='black', linewidth=0.3,
                      error_kw={'linewidth': 0.5, 'capthick': 0.5})
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=5.5)
        ax.set_title(data['info']['label'], fontweight='bold', fontsize=8, pad=3)
        ax.set_ylabel('Mean value', fontsize=7)
        ax.tick_params(axis='both', labelsize=6)

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save_fig(fig, 'fig2_modality_band_statistics')


# ============================================================================
# FIGURE 3: M3 Modality Ablation — R² and MAE comparison (bar chart)
# ============================================================================
def fig3_m3_modality_ablation():
    """M3 ablation study: R² and MAE across modality combinations, per crop and overall."""
    print('Figure 3: M3 modality ablation bar charts...')

    pred_dir = 'M3/predictions'
    csv_files = sorted(Path(pred_dir).glob('*.csv'))

    # Compute metrics per crop per modality
    results = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        y_true = df['YieldGT'].values
        y_pred = df['Prediction'].values
        metrics = calculate_metrics(y_true, y_pred)
        modal = extract_modal_code(csv_file.stem)
        crop = extract_crop(csv_file.stem)
        results.append({'Modal': modal, 'Crop': crop, **metrics})

    rdf = pd.DataFrame(results)

    # Compute overall (combined corn + soybean) per modality
    overall = []
    for modal in rdf['Modal'].unique():
        sub = rdf[rdf['Modal'] == modal]
        if len(sub) == 0:
            continue
        # Combine all predictions
        all_true, all_pred = [], []
        for _, row_csv in enumerate(csv_files):
            if extract_modal_code(row_csv.stem) == modal:
                d = pd.read_csv(row_csv)
                all_true.extend(d['YieldGT'].values)
                all_pred.extend(d['Prediction'].values)
        met = calculate_metrics(np.array(all_true), np.array(all_pred))
        overall.append({'Modal': modal, **met})

    odf = pd.DataFrame(overall).sort_values('R2', ascending=False)

    # ---- Figure: 3 panels (Corn R², Soybean R², Overall R²+MAE) ----
    fig = plt.figure(figsize=(DOUBLE_COL, 5.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel (a): Corn R²
    ax_a = fig.add_subplot(gs[0, 0])
    corn_df = rdf[rdf['Crop'] == 'Corn'].sort_values('R2', ascending=True)
    y_pos = np.arange(len(corn_df))
    ax_a.barh(y_pos, corn_df['R2'], color=CORN_COLOR, edgecolor='black',
              linewidth=0.3, height=0.7, alpha=0.9)
    for i, (_, row) in enumerate(corn_df.iterrows()):
        ax_a.text(row['R2'] + 0.01, i, f"{row['R2']:.3f}", va='center', fontsize=6)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(corn_df['Modal'])
    ax_a.set_xlabel('$R^2$')
    ax_a.set_title('(a) Corn — $R^2$ by Modality', fontweight='bold', fontsize=8)
    ax_a.set_xlim(0, corn_df['R2'].max() * 1.2)

    # Panel (b): Soybean R²
    ax_b = fig.add_subplot(gs[0, 1])
    soy_df = rdf[rdf['Crop'] == 'Soybean'].sort_values('R2', ascending=True)
    y_pos = np.arange(len(soy_df))
    ax_b.barh(y_pos, soy_df['R2'], color=SOYBEAN_COLOR, edgecolor='black',
              linewidth=0.3, height=0.7, alpha=0.9)
    for i, (_, row) in enumerate(soy_df.iterrows()):
        ax_b.text(row['R2'] + 0.01, i, f"{row['R2']:.3f}", va='center', fontsize=6)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(soy_df['Modal'])
    ax_b.set_xlabel('$R^2$')
    ax_b.set_title('(b) Soybean — $R^2$ by Modality', fontweight='bold', fontsize=8)
    ax_b.set_xlim(0, soy_df['R2'].max() * 1.2)

    # Panel (c): Overall R² + MAE grouped bar
    ax_c = fig.add_subplot(gs[1, :])
    odf_sorted = odf.sort_values('R2', ascending=False)
    x = np.arange(len(odf_sorted))
    w = 0.35
    bars1 = ax_c.bar(x - w/2, odf_sorted['R2'], w, color=NEUTRAL_BLUE, alpha=0.85,
                     edgecolor='black', linewidth=0.3, label='$R^2$')
    ax_c2 = ax_c.twinx()
    bars2 = ax_c2.bar(x + w/2, odf_sorted['MAE'], w, color=ACCENT_RED, alpha=0.7,
                      edgecolor='black', linewidth=0.3, label='MAE')

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(odf_sorted['Modal'], rotation=30, ha='right')
    ax_c.set_ylabel('$R^2$ Score', color=NEUTRAL_BLUE)
    ax_c2.set_ylabel('MAE (bu/acre)', color=ACCENT_RED)
    ax_c.set_title('(c) Overall — $R^2$ and MAE by Modality Configuration', fontweight='bold', fontsize=8)

    # Combined legend
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7,
                framealpha=0.9, edgecolor='gray')

    # Add value labels on bars
    for bar, val in zip(bars1, odf_sorted['R2']):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f'{val:.2f}', ha='center', va='bottom', fontsize=5.5, color=NEUTRAL_BLUE)
    for bar, val in zip(bars2, odf_sorted['MAE']):
        ax_c2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=5.5, color=ACCENT_RED)

    ax_c2.spines['right'].set_visible(True)
    save_fig(fig, 'fig3_m3_modality_ablation')


# ============================================================================
# FIGURE 4: M3 Scatter plots — Best configurations (2x2 grid)
# ============================================================================
def fig4_m3_scatter_best():
    """R² scatter plots for the best M3 modality configurations."""
    print('Figure 4: M3 best config scatter plots...')

    pred_dir = 'M3/predictions'
    csv_files = sorted(Path(pred_dir).glob('*.csv'))

    # Compute overall metrics and find best
    modal_data = {}
    for csv_file in csv_files:
        modal = extract_modal_code(csv_file.stem)
        crop = extract_crop(csv_file.stem)
        df = pd.read_csv(csv_file)
        key = (modal, crop)
        modal_data[key] = df

    # Select best overall + best per crop
    # Use: S12dw (best reported), S12cdw, S12d, S12 as the 4 panels
    target_configs = [
        ('S12dw', 'Corn'), ('S12dw', 'Soybean'),
        ('S12cdw', 'Corn'), ('S12cdw', 'Soybean'),
    ]

    # Fallback: find available configs
    available = list(modal_data.keys())
    configs_to_plot = []
    for tc in target_configs:
        if tc in available:
            configs_to_plot.append(tc)
    if len(configs_to_plot) < 4:
        # Fill with other available configs
        for k in available:
            if k not in configs_to_plot:
                configs_to_plot.append(k)
            if len(configs_to_plot) >= 4:
                break

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL))
    axes = axes.flatten()

    for idx, (modal, crop) in enumerate(configs_to_plot[:4]):
        ax = axes[idx]
        df = modal_data[(modal, crop)]
        y_true = df['YieldGT'].values
        y_pred = df['Prediction'].values
        metrics = calculate_metrics(y_true, y_pred)

        color = CORN_COLOR if crop == 'Corn' else SOYBEAN_COLOR

        ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='black',
                   linewidths=0.4, color=color, zorder=3)

        # 1:1 line
        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5, zorder=2, label='1:1')

        # Regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, p(x_fit), color=ACCENT_RED, linewidth=1.0, alpha=0.8,
                zorder=2, label=f'y={z[0]:.2f}x+{z[1]:.1f}')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

        # Metrics box
        txt = (f'$R^2$ = {metrics["R2"]:.3f}\n'
               f'MAE = {metrics["MAE"]:.1f} bu/ac\n'
               f'RMSE = {metrics["RMSE"]:.1f} bu/ac\n'
               f'n = {metrics["N"]}')
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=6.5,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9,
                          ec='gray', lw=0.4))

        letter = chr(97 + idx)
        ax.set_title(f'({letter}) {modal} — {crop}', fontweight='bold', fontsize=8, pad=3)
        ax.set_xlabel('Observed Yield (bu/acre)', fontsize=7.5)
        ax.set_ylabel('Predicted Yield (bu/acre)', fontsize=7.5)
        ax.legend(loc='lower right', fontsize=6, framealpha=0.9)
        ax.grid(True, alpha=0.2, linewidth=0.3, zorder=1)

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save_fig(fig, 'fig4_m3_scatter_best')


# ============================================================================
# FIGURE 5: M3 ALL scatter plots — grid of all modality configs
# ============================================================================
def fig5_m3_scatter_all():
    """R² scatter plots for ALL M3 modality configurations in a large grid."""
    print('Figure 5: M3 all config scatter plots...')

    pred_dir = 'M3/predictions'
    csv_files = sorted(Path(pred_dir).glob('*.csv'))

    n_files = len(csv_files)
    if n_files == 0:
        print('  No prediction CSVs found, skipping.')
        return

    # Arrange in grid
    ncols = 4
    nrows = int(np.ceil(n_files / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, nrows * 1.7))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    for idx, csv_file in enumerate(csv_files):
        ax = axes_flat[idx]
        df = pd.read_csv(csv_file)
        y_true = df['YieldGT'].values
        y_pred = df['Prediction'].values
        metrics = calculate_metrics(y_true, y_pred)
        modal = extract_modal_code(csv_file.stem)
        crop = extract_crop(csv_file.stem)
        color = CORN_COLOR if crop == 'Corn' else SOYBEAN_COLOR

        ax.scatter(y_true, y_pred, alpha=0.5, s=12, edgecolors='none', color=color, zorder=3)

        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.4, zorder=2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

        ax.set_title(f'{modal} {crop}\n$R^2$={metrics["R2"]:.3f}', fontsize=5.5, pad=2)
        ax.tick_params(labelsize=5)

        if idx % ncols == 0:
            ax.set_ylabel('Pred', fontsize=5.5)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel('Obs', fontsize=5.5)

    # Hide unused axes
    for i in range(n_files, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save_fig(fig, 'fig5_m3_scatter_all')


# ============================================================================
# FIGURE 6: Year-wise Performance (Terramind_year.csv)
# ============================================================================
def fig6_yearwise_performance():
    """Year-by-year performance from Terramind_year.csv (M3 best model)."""
    print('Figure 6: Year-wise performance...')

    df = pd.read_csv('Terramind_year.csv')
    df['Exp'] = df['Exp'].astype(str)

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL, 2.8))

    x = np.arange(len(df))
    w = 0.35

    bars1 = ax1.bar(x - w/2, df['R-square'], w, color=NEUTRAL_BLUE, alpha=0.85,
                    edgecolor='black', linewidth=0.4, label='$R^2$', zorder=3)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, df['MAE'], w, color=ACCENT_RED, alpha=0.7,
                    edgecolor='black', linewidth=0.4, label='MAE', zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Exp'])
    ax1.set_xlabel('Test Year')
    ax1.set_ylabel('$R^2$ Score', color=NEUTRAL_BLUE)
    ax2.set_ylabel('MAE (bu/acre)', color=ACCENT_RED)

    # Value labels
    for bar, val in zip(bars1, df['R-square']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=6, color=NEUTRAL_BLUE,
                 fontweight='bold')
    for bar, val in zip(bars2, df['MAE']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=6, color=ACCENT_RED,
                 fontweight='bold')

    # Combined legend
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc='upper left', fontsize=7, framealpha=0.9)

    ax1.set_ylim(0, df['R-square'].max() * 1.25)
    ax2.set_ylim(0, df['MAE'].max() * 1.25)
    ax1.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)
    ax2.spines['right'].set_visible(True)

    fig.tight_layout()
    save_fig(fig, 'fig6_yearwise_performance')


# ============================================================================
# FIGURE 7: Classical ML Temporal (Weekly) — R² progression by week
# ============================================================================
def fig7_classical_ml_temporal():
    """Classical ML weekly R² progression (XGBoost + PLSR) over 24 weeks."""
    print('Figure 7: Classical ML weekly temporal progression...')

    df = pd.read_csv('classical_ml_results_s12wd.csv')

    # Extract week number
    df['week'] = df['biweek'].str.extract(r'weekly_(\d+)').astype(int)
    df = df.sort_values('week')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8), sharey=False)

    # Panel (a): Test R²
    for model, color, marker in [('XGBoost', NEUTRAL_BLUE, 'o'), ('PLSR', ACCENT_RED, 's')]:
        sub = df[df['model'] == model]
        ax1.plot(sub['week'], sub['test_r2'], color=color, marker=marker,
                 markersize=3, linewidth=0.8, label=model, zorder=3)
        ax1.fill_between(sub['week'], sub['test_r2'], alpha=0.1, color=color)

    ax1.set_xlabel('Week')
    ax1.set_ylabel('Test $R^2$')
    ax1.set_title('(a) Test $R^2$ by Week', fontweight='bold', fontsize=8)
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.2, linewidth=0.3)
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    # Panel (b): Test MAE
    for model, color, marker in [('XGBoost', NEUTRAL_BLUE, 'o'), ('PLSR', ACCENT_RED, 's')]:
        sub = df[df['model'] == model]
        ax2.plot(sub['week'], sub['test_mae'], color=color, marker=marker,
                 markersize=3, linewidth=0.8, label=model, zorder=3)

    ax2.set_xlabel('Week')
    ax2.set_ylabel('Test MAE')
    ax2.set_title('(b) Test MAE by Week', fontweight='bold', fontsize=8)
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.2, linewidth=0.3)

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, 'fig7_classical_ml_temporal_weekly')


# ============================================================================
# FIGURE 8: Classical ML vs TerraMind Comparison
# ============================================================================
def fig8_classical_vs_terramind():
    """Compare classical ML baselines with TerraMind deep learning models."""
    print('Figure 8: Classical ML vs TerraMind comparison...')

    # Classical ML best results (from classical_ml_summary_s12wd_biweek12.csv — the summary of best)
    classical = pd.read_csv('classical_ml_summary_s12wd_biweek12.csv')

    # TerraMind M3 overall metrics
    m3_metrics = pd.read_csv('M3/plots_stat/overall_metrics.csv')
    # Pick the best M3 (S12cdw has R2=0.969, but this is train. Use Modality_Config for test metrics).
    # Actually we ignore Modality_Config per user instructions. Use M3 prediction CSVs directly.

    # Compute M3 test metrics from predictions
    pred_dir = 'M3/predictions'
    m3_test = {}
    for csv_file in sorted(Path(pred_dir).glob('*.csv')):
        modal = extract_modal_code(csv_file.stem)
        df = pd.read_csv(csv_file)
        if modal not in m3_test:
            m3_test[modal] = {'true': [], 'pred': []}
        m3_test[modal]['true'].extend(df['YieldGT'].values)
        m3_test[modal]['pred'].extend(df['Prediction'].values)

    # Get best M3 config
    best_m3_modal, best_m3_r2 = None, -1
    m3_results = {}
    for modal, data in m3_test.items():
        yt = np.array(data['true'])
        yp = np.array(data['pred'])
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        m3_results[modal] = {'R2': r2, 'MAE': mae}
        if r2 > best_m3_r2:
            best_m3_r2 = r2
            best_m3_modal = modal

    # Build comparison data
    # Note: classical ML values are normalized (0-1 range), TerraMind is in bu/acre
    # We'll show R² which is scale-invariant
    models = ['PLSR', 'XGBoost']
    classical_r2 = [classical[classical['model'] == m]['test_r2'].values[0] for m in models]
    classical_mae = [classical[classical['model'] == m]['test_mae'].values[0] for m in models]

    # Add top M3 configs
    sorted_m3 = sorted(m3_results.items(), key=lambda x: x[1]['R2'], reverse=True)[:3]
    for modal, met in sorted_m3:
        models.append(f'TM-{modal}')
        classical_r2.append(met['R2'])
        classical_mae.append(met['MAE'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    colors = [ACCENT_RED, NEUTRAL_BLUE] + [MODEL_COLORS['M3']] * len(sorted_m3)
    x = np.arange(len(models))

    # R² comparison
    bars = ax1.bar(x, classical_r2, color=colors, edgecolor='black', linewidth=0.4,
                   alpha=0.85, zorder=3)
    for bar, val in zip(bars, classical_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha='right', fontsize=7)
    ax1.set_ylabel('Test $R^2$')
    ax1.set_title('(a) $R^2$ Comparison', fontweight='bold', fontsize=8)
    ax1.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)
    ax1.set_ylim(0, max(classical_r2) * 1.2)

    # MAE comparison — note: classical MAE is normalized, TerraMind is bu/acre
    # Only compare R² fairly. For MAE, plot separately with note
    ax2.bar(x, classical_mae, color=colors, edgecolor='black', linewidth=0.4,
            alpha=0.85, zorder=3)
    for i, (bar_x, val) in enumerate(zip(x, classical_mae)):
        unit = '' if i < 2 else ' bu/ac'
        ax2.text(bar_x, val + max(classical_mae)*0.02,
                 f'{val:.2f}{unit}', ha='center', va='bottom', fontsize=5.5, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha='right', fontsize=7)
    ax2.set_ylabel('Test MAE')
    ax2.set_title('(b) MAE Comparison', fontweight='bold', fontsize=8)
    ax2.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)

    # Add note about scale difference
    ax2.text(0.5, 0.92, 'Note: Classical ML uses normalized yield;\nTerraMind uses bu/acre',
             transform=ax2.transAxes, fontsize=5.5, ha='center', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='orange', lw=0.4))

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, 'fig8_classical_vs_terramind')


# ============================================================================
# FIGURE 9: M4 Temporal Ablation — Performance vs observation window
# ============================================================================
def fig9_m4_temporal_ablation():
    """M4: Performance across different temporal windows (weekly).
    Uses Terramind_test.csv which has S12wd_1 through S12wd_12 weekly results."""
    print('Figure 9: M4 temporal ablation...')

    # Read the temporal results file (previously found as Terramind_test.csv in old_results)
    # Actually those are in the wandb runs named S12wd_N
    # Parse from wandb summary files
    temporal_data = []
    wandb_dir = 'wandb'
    for run_dir in sorted(os.listdir(wandb_dir)):
        if not run_dir.startswith('run-'):
            continue
        # Check if run name contains S12wd_N pattern
        summary_path = os.path.join(wandb_dir, run_dir, 'files', 'wandb-summary.json')
        config_path = os.path.join(wandb_dir, run_dir, 'files', 'config.yaml')
        if not os.path.exists(summary_path):
            continue

        # Check run name (from directory name)
        parts = run_dir.split('-')
        if len(parts) >= 3:
            run_name = '-'.join(parts[2:])
        else:
            continue

        # Only process S12wd_N runs (weekly temporal)
        import re
        match = re.match(r'S12wd_(\d+)$', run_name)
        if not match:
            continue

        week = int(match.group(1))

        import json
        with open(summary_path) as f:
            summary = json.load(f)

        val_r2 = summary.get('val/R2_Score')
        val_mae = summary.get('val/MAE')
        if val_r2 is not None and val_mae is not None:
            temporal_data.append({
                'week': week,
                'val_R2': val_r2,
                'val_MAE': val_mae,
                'run': run_dir,
            })

    if not temporal_data:
        print('  No temporal data found, skipping.')
        return

    tdf = pd.DataFrame(temporal_data)
    # Take the best run per week (highest val_R2)
    tdf_best = tdf.loc[tdf.groupby('week')['val_R2'].idxmax()].sort_values('week')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    # R² vs week
    ax1.plot(tdf_best['week'], tdf_best['val_R2'], color=MODEL_COLORS['M4'],
             marker='o', markersize=4, linewidth=1.2, zorder=3)
    ax1.fill_between(tdf_best['week'], tdf_best['val_R2'], alpha=0.15,
                     color=MODEL_COLORS['M4'])
    ax1.set_xlabel('Observation Window (weeks)')
    ax1.set_ylabel('Validation $R^2$')
    ax1.set_title('(a) $R^2$ vs Temporal Window', fontweight='bold', fontsize=8)
    ax1.grid(True, alpha=0.2, linewidth=0.3)
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    # Annotate key points
    best_row = tdf_best.loc[tdf_best['val_R2'].idxmax()]
    ax1.annotate(f"Best: wk {int(best_row['week'])}\n$R^2$={best_row['val_R2']:.3f}",
                 xy=(best_row['week'], best_row['val_R2']),
                 xytext=(best_row['week']-2, best_row['val_R2']-0.1),
                 fontsize=6, arrowprops=dict(arrowstyle='->', color='black', lw=0.6),
                 bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='gray', lw=0.4))

    # MAE vs week
    ax2.plot(tdf_best['week'], tdf_best['val_MAE'], color=ACCENT_RED,
             marker='s', markersize=4, linewidth=1.2, zorder=3)
    ax2.fill_between(tdf_best['week'], tdf_best['val_MAE'], alpha=0.15,
                     color=ACCENT_RED)
    ax2.set_xlabel('Observation Window (weeks)')
    ax2.set_ylabel('Validation MAE')
    ax2.set_title('(b) MAE vs Temporal Window', fontweight='bold', fontsize=8)
    ax2.grid(True, alpha=0.2, linewidth=0.3)

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, 'fig9_m4_temporal_ablation')


# ============================================================================
# FIGURE 10: M5 — Different normalization comparison
# ============================================================================
def fig10_m5_normalization():
    """M5: Different normalization approach results."""
    print('Figure 10: M5 different normalization...')

    import json

    # Extract M5 metrics from WandB
    m5_runs = []
    m5_wandb = 'M5/wandb'
    if not os.path.exists(m5_wandb):
        print('  M5 WandB not found, skipping.')
        return

    for run_dir in sorted(os.listdir(m5_wandb)):
        if not run_dir.startswith('run-'):
            continue
        summary_path = os.path.join(m5_wandb, run_dir, 'files', 'wandb-summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        val_r2 = summary.get('val/R2_Score')
        val_mae = summary.get('val/MAE')
        if val_r2 is not None:
            # Determine config from M5 output structure
            # M5 corn: S12D_8, soybean: S12D_24
            m5_runs.append({
                'run': run_dir,
                'val_R2': val_r2,
                'val_MAE': val_mae,
                'val_RMSE': summary.get('val/RMSE'),
                'train_R2': summary.get('train/R2_Score'),
            })

    if not m5_runs:
        print('  No M5 results found, skipping.')
        return

    m5df = pd.DataFrame(m5_runs)

    # Compare M5 with M3 best (S12dw results)
    # M5 uses S12d with different normalization
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))

    labels = [f'M5-Run{i+1}' for i in range(len(m5df))]
    colors = [MODEL_COLORS['M5']] * len(m5df)
    x = np.arange(len(m5df))

    ax.bar(x, m5df['val_R2'], color=colors, edgecolor='black', linewidth=0.4,
           alpha=0.85, zorder=3)
    for i, val in enumerate(m5df['val_R2']):
        ax.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Validation $R^2$')
    ax.set_title('M5: Different Normalization Experiments', fontweight='bold', fontsize=8)
    ax.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)

    fig.tight_layout()
    save_fig(fig, 'fig10_m5_normalization')


# ============================================================================
# FIGURE 11: Comprehensive Model Comparison (M1–M5 summary)
# ============================================================================
def fig11_model_summary():
    """Summary comparison across all 5 model experiments."""
    print('Figure 11: Model summary M1–M5...')

    import json

    # ---- Collect results from each model ----

    # M3: from prediction CSVs (best overall config)
    pred_dir = 'M3/predictions'
    m3_test = {}
    for csv_file in sorted(Path(pred_dir).glob('*.csv')):
        modal = extract_modal_code(csv_file.stem)
        df = pd.read_csv(csv_file)
        if modal not in m3_test:
            m3_test[modal] = {'true': [], 'pred': []}
        m3_test[modal]['true'].extend(df['YieldGT'].values)
        m3_test[modal]['pred'].extend(df['Prediction'].values)

    best_m3 = max(m3_test.items(),
                  key=lambda x: r2_score(np.array(x[1]['true']), np.array(x[1]['pred'])))
    m3_r2 = r2_score(np.array(best_m3[1]['true']), np.array(best_m3[1]['pred']))
    m3_mae = mean_absolute_error(np.array(best_m3[1]['true']), np.array(best_m3[1]['pred']))

    # M4: from wandb (best weekly temporal)
    temporal_data = []
    import re
    for run_dir in sorted(os.listdir('wandb')):
        if not run_dir.startswith('run-'):
            continue
        parts = run_dir.split('-')
        run_name = '-'.join(parts[2:]) if len(parts) >= 3 else ''
        match = re.match(r'S12wd_(\d+)$', run_name)
        if not match:
            continue
        summary_path = os.path.join('wandb', run_dir, 'files', 'wandb-summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            s = json.load(f)
        val_r2 = s.get('val/R2_Score')
        val_mae = s.get('val/MAE')
        if val_r2 is not None:
            temporal_data.append({'week': int(match.group(1)), 'R2': val_r2, 'MAE': val_mae})

    m4_r2 = max(t['R2'] for t in temporal_data) if temporal_data else 0
    m4_mae = min(t['MAE'] for t in temporal_data) if temporal_data else 0

    # M5: from wandb
    m5_r2, m5_mae = 0, 0
    for run_dir in sorted(os.listdir('M5/wandb')):
        if not run_dir.startswith('run-'):
            continue
        summary_path = os.path.join('M5/wandb', run_dir, 'files', 'wandb-summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            s = json.load(f)
        val_r2 = s.get('val/R2_Score', 0)
        if val_r2 and val_r2 > m5_r2:
            m5_r2 = val_r2
            m5_mae = s.get('val/MAE', 0)

    # M1 (frozen backbone, S12d) and M2 (unfrozen, S12wdsc) — extract from configs
    # M1 and M2 don't have separate wandb dirs; they used the main predictions/ dir
    # M1 = S12d config (frozen backbone), M2 = S12wdsc (unfrozen, all modalities)
    # Use predictions/ for M1/M2
    m1_true, m1_pred = [], []
    m2_true, m2_pred = [], []
    for csv_file in sorted(Path('predictions').glob('*.csv')):
        modal = extract_modal_code(csv_file.stem)
        df = pd.read_csv(csv_file)
        if modal == 'S12d':
            m1_true.extend(df['YieldGT'].values)
            m1_pred.extend(df['Prediction'].values)
        # S12cdws or similar all-modality config for M2
        if modal in ['S12cdws', 'S12cdw']:
            m2_true.extend(df['YieldGT'].values)
            m2_pred.extend(df['Prediction'].values)

    m1_r2 = r2_score(np.array(m1_true), np.array(m1_pred)) if m1_true else 0
    m1_mae = mean_absolute_error(np.array(m1_true), np.array(m1_pred)) if m1_true else 0
    m2_r2 = r2_score(np.array(m2_true), np.array(m2_pred)) if m2_true else 0
    m2_mae = mean_absolute_error(np.array(m2_true), np.array(m2_pred)) if m2_true else 0

    # Build summary table
    summary = pd.DataFrame([
        {'Model': 'M1', 'Config': 'S12d (frozen)', 'R2': m1_r2, 'MAE': m1_mae,
         'Description': 'Frozen backbone\nS2+S1+DEM'},
        {'Model': 'M2', 'Config': 'S12cdw (unfrozen)', 'R2': m2_r2, 'MAE': m2_mae,
         'Description': 'Unfrozen backbone\nAll modalities'},
        {'Model': 'M3', 'Config': f'{best_m3[0]} (ablation)', 'R2': m3_r2, 'MAE': m3_mae,
         'Description': 'Modality ablation\nBest combination'},
        {'Model': 'M4', 'Config': 'S12wd (temporal)', 'R2': m4_r2, 'MAE': m4_mae,
         'Description': 'Temporal ablation\nBest window'},
        {'Model': 'M5', 'Config': 'S12d (diff norm)', 'R2': m5_r2, 'MAE': m5_mae,
         'Description': 'Different norm\nExploration'},
    ])

    # ---- Figure: grouped horizontal bar ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8), sharey=True)

    y = np.arange(len(summary))
    colors = [MODEL_COLORS[m] for m in summary['Model']]

    # R² bars
    bars1 = ax1.barh(y, summary['R2'], color=colors, edgecolor='black',
                     linewidth=0.4, height=0.6, alpha=0.9, zorder=3)
    for i, val in enumerate(summary['R2']):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=7, fontweight='bold')
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{row['Model']}\n{row['Config']}" for _, row in summary.iterrows()],
                        fontsize=6.5)
    ax1.set_xlabel('$R^2$ Score')
    ax1.set_title('(a) $R^2$ Comparison', fontweight='bold', fontsize=8)
    ax1.grid(axis='x', alpha=0.2, linewidth=0.3, zorder=0)
    ax1.set_xlim(0, summary['R2'].max() * 1.2)

    # MAE bars
    bars2 = ax2.barh(y, summary['MAE'], color=colors, edgecolor='black',
                     linewidth=0.4, height=0.6, alpha=0.9, zorder=3)
    for i, val in enumerate(summary['MAE']):
        label = f'{val:.2f}' if val < 1 else f'{val:.1f}'
        ax2.text(val + max(summary['MAE'])*0.02, i, label, va='center', fontsize=7, fontweight='bold')
    ax2.set_xlabel('MAE')
    ax2.set_title('(b) MAE Comparison', fontweight='bold', fontsize=8)
    ax2.grid(axis='x', alpha=0.2, linewidth=0.3, zorder=0)

    fig.tight_layout(w_pad=1.5)
    save_fig(fig, 'fig11_model_summary_m1_m5')


# ============================================================================
# FIGURE 12: Weather band distributions
# ============================================================================
def fig12_weather_distributions():
    """Weather variable distributions across all processed data."""
    print('Figure 12: Weather band distributions...')

    base = 'processed_data/weekly_24/processed_data_weekly_24/WEATHER'
    band_names = ['DAYL', 'PRCP', 'SRAD', 'SWE', 'TMAX', 'TMIN', 'VP']

    # Sample files for efficient computation
    all_files = sorted(os.listdir(base))
    sample_files = all_files[::max(1, len(all_files)//50)]  # ~50 files

    band_data = {name: [] for name in band_names}

    for fname in sample_files:
        fpath = os.path.join(base, fname)
        arr = np.load(fpath)  # (24, 7, 224, 224)
        for b, name in enumerate(band_names):
            if b < arr.shape[1]:
                vals = arr[:, b].flatten()
                valid = vals[~np.isnan(vals)]
                # Subsample to keep memory manageable
                if len(valid) > 5000:
                    valid = np.random.choice(valid, 5000, replace=False)
                band_data[name].extend(valid.tolist())

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL, 3.8))
    axes = axes.flatten()
    colors = sns.color_palette('Set2', len(band_names))

    for idx, (name, vals) in enumerate(band_data.items()):
        ax = axes[idx]
        vals = np.array(vals)
        ax.hist(vals, bins=60, color=colors[idx], alpha=0.8, edgecolor='white',
                linewidth=0.2, density=True)
        ax.set_title(name, fontweight='bold', fontsize=7.5, pad=2)
        ax.tick_params(labelsize=5.5)
        txt = f'$\\mu$={vals.mean():.1f}\n$\\sigma$={vals.std():.1f}'
        ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=5.5,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray', lw=0.3))

    # Hide last unused axis
    axes[7].set_visible(False)

    fig.suptitle('Weather Band Distributions (Weekly Data)', fontweight='bold', fontsize=9, y=1.01)
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    save_fig(fig, 'fig12_weather_distributions')


# ============================================================================
# FIGURE 13: Sentinel-2 NDVI temporal profile
# ============================================================================
def fig13_ndvi_temporal():
    """Show NDVI temporal profiles from S2 data across the growing season."""
    print('Figure 13: NDVI temporal profiles...')

    base = 'processed_data/weekly_24/processed_data_weekly_24/S2L2A'
    all_files = sorted(os.listdir(base))

    corn_files = [f for f in all_files if 'Corn' in f][:20]
    soy_files = [f for f in all_files if 'Soybean' in f][:20]

    def compute_ndvi_profile(files):
        profiles = []
        for fname in files:
            arr = np.load(os.path.join(base, fname))  # (24, 12, 224, 224)
            # NIR ~ band 6 (B8 at index 6), Red ~ band 2 (B4 at index 2)
            nir = arr[:, 6, :, :].astype(float)
            red = arr[:, 2, :, :].astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red + 1e-8)
            weekly_mean = np.nanmean(ndvi, axis=(1, 2))  # (24,)
            profiles.append(weekly_mean)
        return np.array(profiles)

    corn_ndvi = compute_ndvi_profile(corn_files)
    soy_ndvi = compute_ndvi_profile(soy_files)

    weeks = np.arange(1, 25)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))

    # Mean ± std
    for data, label, color in [(corn_ndvi, 'Corn', CORN_COLOR), (soy_ndvi, 'Soybean', SOYBEAN_COLOR)]:
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        ax.plot(weeks, mean, color=color, linewidth=1.2, label=label, zorder=3)
        ax.fill_between(weeks, mean - std, mean + std, alpha=0.2, color=color, zorder=2)

    ax.set_xlabel('Week')
    ax.set_ylabel('Mean NDVI')
    ax.set_title('NDVI Temporal Profile', fontweight='bold', fontsize=8)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.3, zorder=0)
    ax.set_xlim(1, 24)

    fig.tight_layout()
    save_fig(fig, 'fig13_ndvi_temporal_profile')


# ============================================================================
# FIGURE 14: Data split summary and sample counts
# ============================================================================
def fig14_data_splits():
    """Visualize train/val/test split counts and year distribution."""
    print('Figure 14: Data split summary...')

    base = 'processed_data/weekly_24/processed_data_weekly_24'
    splits = {}
    for split_file in ['train_corn.txt', 'val_corn.txt', 'test_corn.txt',
                       'train_soybean.txt', 'val_soybean.txt', 'test_soybean.txt']:
        fpath = os.path.join(base, split_file)
        if os.path.exists(fpath):
            with open(fpath) as f:
                names = [l.strip() for l in f if l.strip()]
            parts = split_file.replace('.txt', '').split('_')
            split_name = parts[0].capitalize()
            crop = parts[1].capitalize()
            splits[(split_name, crop)] = names

    # Count by year
    year_counts = {}
    for (split, crop), names in splits.items():
        for name in names:
            # Extract year from name like ST2019IA0148
            import re
            m = re.search(r'ST(\d{4})', name)
            if m:
                yr = m.group(1)
                key = (split, crop, yr)
                year_counts[key] = year_counts.get(key, 0) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    # Panel (a): Split counts
    split_summary = {}
    for (split, crop), names in splits.items():
        if split not in split_summary:
            split_summary[split] = {}
        split_summary[split][crop] = len(names)

    x = np.arange(len(split_summary))
    w = 0.35
    splits_ordered = ['Train', 'Val', 'Test']
    corn_counts = [split_summary.get(s, {}).get('Corn', 0) for s in splits_ordered]
    soy_counts = [split_summary.get(s, {}).get('Soybean', 0) for s in splits_ordered]

    ax1.bar(x - w/2, corn_counts, w, color=CORN_COLOR, edgecolor='black',
            linewidth=0.4, label='Corn', alpha=0.85)
    ax1.bar(x + w/2, soy_counts, w, color=SOYBEAN_COLOR, edgecolor='black',
            linewidth=0.4, label='Soybean', alpha=0.85)

    for i, (c, s) in enumerate(zip(corn_counts, soy_counts)):
        ax1.text(i - w/2, c + 1, str(c), ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        ax1.text(i + w/2, s + 1, str(s), ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(splits_ordered)
    ax1.set_ylabel('Number of Fields')
    ax1.set_title('(a) Train/Val/Test Split', fontweight='bold', fontsize=8)
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)

    # Panel (b): Year distribution in training set
    years = sorted(set(yr for (_, _, yr) in year_counts.keys()))
    corn_by_year = [sum(year_counts.get(('Train', 'Corn', yr), 0) for _ in [1]) for yr in years]
    soy_by_year = [sum(year_counts.get(('Train', 'Soybean', yr), 0) for _ in [1]) for yr in years]

    x2 = np.arange(len(years))
    ax2.bar(x2 - w/2, corn_by_year, w, color=CORN_COLOR, edgecolor='black',
            linewidth=0.4, label='Corn', alpha=0.85)
    ax2.bar(x2 + w/2, soy_by_year, w, color=SOYBEAN_COLOR, edgecolor='black',
            linewidth=0.4, label='Soybean', alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(years, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Number of Fields')
    ax2.set_title('(b) Training Data by Year', fontweight='bold', fontsize=8)
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, 'fig14_data_splits')


# ============================================================================
# FIGURE 15: M3 Modality Contribution Heatmap
# ============================================================================
def fig15_m3_modality_heatmap():
    """Heatmap showing R² for each modality combination (M3)."""
    print('Figure 15: M3 modality contribution heatmap...')

    pred_dir = 'M3/predictions'
    csv_files = sorted(Path(pred_dir).glob('*.csv'))

    # Compute metrics
    results = {}
    for csv_file in csv_files:
        modal = extract_modal_code(csv_file.stem)
        crop = extract_crop(csv_file.stem)
        df = pd.read_csv(csv_file)
        r2 = r2_score(df['YieldGT'], df['Prediction'])
        results[(modal, crop)] = r2

    # Also compute overall
    modal_overall = {}
    for csv_file in csv_files:
        modal = extract_modal_code(csv_file.stem)
        if modal not in modal_overall:
            modal_overall[modal] = {'true': [], 'pred': []}
        df = pd.read_csv(csv_file)
        modal_overall[modal]['true'].extend(df['YieldGT'].values)
        modal_overall[modal]['pred'].extend(df['Prediction'].values)

    for modal in modal_overall:
        yt = np.array(modal_overall[modal]['true'])
        yp = np.array(modal_overall[modal]['pred'])
        results[(modal, 'Overall')] = r2_score(yt, yp)

    # Build matrix
    modalities = sorted(set(m for m, _ in results.keys()))
    crops = ['Corn', 'Soybean', 'Overall']

    matrix = np.full((len(modalities), len(crops)), np.nan)
    for i, mod in enumerate(modalities):
        for j, crop in enumerate(crops):
            if (mod, crop) in results:
                matrix[i, j] = results[(mod, crop)]

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, max(2.5, len(modalities) * 0.35)))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(crops)))
    ax.set_xticklabels(crops, fontsize=7.5)
    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels(modalities, fontsize=7)

    # Annotate cells
    for i in range(len(modalities)):
        for j in range(len(crops)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.3 or val > 0.85 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=6.5, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$R^2$ Score', fontsize=8)
    cbar.ax.tick_params(labelsize=6.5)

    ax.set_title('M3: Modality Ablation $R^2$ Heatmap', fontweight='bold', fontsize=8, pad=8)

    fig.tight_layout()
    save_fig(fig, 'fig15_m3_modality_heatmap')


# ============================================================================
# FIGURE 16: Data sample counts per year (from yield parquets)
# ============================================================================
def fig16_yield_statistics():
    """Yield statistics across years — field-level sample sizes, mean yield, variability."""
    print('Figure 16: Yield statistics by year...')

    years = list(range(2017, 2025))
    stats = []
    for yr in years:
        ppath = f'Yield_{yr}_filtered.parquet'
        if not os.path.exists(ppath):
            stats.append({'Year': yr, 'n_total': 0, 'n_corn': 0, 'n_soy': 0,
                          'mean_corn': 0, 'std_corn': 0, 'mean_soy': 0, 'std_soy': 0})
            continue
        df = pd.read_parquet(ppath)
        # Aggregate pixels to field-level mean yield
        field_df = df.groupby(['Layer_ID', 'Crop']).agg(
            mean_yield=('Yield', 'mean')).reset_index()

        corn = field_df[field_df['Crop'].str.contains('Corn', case=False, na=False)]['mean_yield']
        corn = corn[(corn > 10) & (corn < 400)]
        soy = field_df[field_df['Crop'].str.contains('Soy', case=False, na=False)]['mean_yield']
        soy = soy[(soy > 5) & (soy < 150)]

        stats.append({
            'Year': yr,
            'n_total': len(corn) + len(soy),
            'n_corn': len(corn),
            'n_soy': len(soy),
            'mean_corn': corn.mean() if len(corn) > 0 else 0,
            'std_corn': corn.std() if len(corn) > 0 else 0,
            'mean_soy': soy.mean() if len(soy) > 0 else 0,
            'std_soy': soy.std() if len(soy) > 0 else 0,
        })

    sdf = pd.DataFrame(stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    x = np.arange(len(sdf))
    w = 0.35

    # Panel (a): Mean field yield ± std
    ax1.bar(x - w/2, sdf['mean_corn'], w, yerr=sdf['std_corn'],
            color=CORN_COLOR, edgecolor='black', linewidth=0.4,
            capsize=2, error_kw={'linewidth': 0.5}, label='Corn', alpha=0.85)
    ax1.bar(x + w/2, sdf['mean_soy'], w, yerr=sdf['std_soy'],
            color=SOYBEAN_COLOR, edgecolor='black', linewidth=0.4,
            capsize=2, error_kw={'linewidth': 0.5}, label='Soybean', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sdf['Year'].astype(str), rotation=45, ha='right', fontsize=7)
    ax1.set_ylabel('Mean Field Yield (bu/acre)')
    ax1.set_title('(a) Mean Field Yield by Year and Crop', fontweight='bold', fontsize=8)
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)

    # Panel (b): Number of fields
    ax2.bar(x - w/2, sdf['n_corn'], w, color=CORN_COLOR, edgecolor='black',
            linewidth=0.4, label='Corn', alpha=0.85)
    ax2.bar(x + w/2, sdf['n_soy'], w, color=SOYBEAN_COLOR, edgecolor='black',
            linewidth=0.4, label='Soybean', alpha=0.85)
    for i, (c, s) in enumerate(zip(sdf['n_corn'], sdf['n_soy'])):
        ax2.text(i - w/2, c + 0.5, str(c), ha='center', va='bottom', fontsize=6, fontweight='bold')
        ax2.text(i + w/2, s + 0.5, str(s), ha='center', va='bottom', fontsize=6, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sdf['Year'].astype(str), rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Number of Fields')
    ax2.set_title('(b) Number of Fields by Year', fontweight='bold', fontsize=8)
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, 'fig16_yield_statistics')


# ============================================================================
# FIGURE 17: Residual analysis for best M3 model
# ============================================================================
def fig17_residual_analysis():
    """Residual analysis for the best M3 configuration."""
    print('Figure 17: Residual analysis...')

    pred_dir = 'M3/predictions'
    csv_files = sorted(Path(pred_dir).glob('*.csv'))

    # Get best overall config
    modal_data = {}
    for csv_file in csv_files:
        modal = extract_modal_code(csv_file.stem)
        df = pd.read_csv(csv_file)
        if modal not in modal_data:
            modal_data[modal] = {'true': [], 'pred': [], 'crop': []}
        crop = extract_crop(csv_file.stem)
        modal_data[modal]['true'].extend(df['YieldGT'].values)
        modal_data[modal]['pred'].extend(df['Prediction'].values)
        modal_data[modal]['crop'].extend([crop] * len(df))

    best_modal = max(modal_data.items(),
                     key=lambda x: r2_score(np.array(x[1]['true']), np.array(x[1]['pred'])))
    modal_name = best_modal[0]
    y_true = np.array(best_modal[1]['true'])
    y_pred = np.array(best_modal[1]['pred'])
    crops = np.array(best_modal[1]['crop'])
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.5))

    # (a) Residual vs predicted
    ax = axes[0]
    for crop, color in [('Corn', CORN_COLOR), ('Soybean', SOYBEAN_COLOR)]:
        mask = crops == crop
        ax.scatter(y_pred[mask], residuals[mask], alpha=0.6, s=20, color=color,
                   edgecolors='black', linewidths=0.3, label=crop, zorder=3)
    ax.axhline(y=0, color='black', linewidth=0.6, linestyle='--', zorder=2)
    ax.set_xlabel('Predicted (bu/acre)', fontsize=7.5)
    ax.set_ylabel('Residual (bu/acre)', fontsize=7.5)
    ax.set_title(f'(a) Residual vs Predicted', fontweight='bold', fontsize=7.5)
    ax.legend(fontsize=6, framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.3, zorder=0)

    # (b) Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=30, color=NEUTRAL_BLUE, alpha=0.8, edgecolor='white',
            linewidth=0.3, density=True)
    ax.axvline(x=0, color='black', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Residual (bu/acre)', fontsize=7.5)
    ax.set_ylabel('Density', fontsize=7.5)
    ax.set_title('(b) Residual Distribution', fontweight='bold', fontsize=7.5)
    txt = f'$\\mu$={np.mean(residuals):.1f}\n$\\sigma$={np.std(residuals):.1f}'
    ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=6.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray', lw=0.3))
    ax.grid(True, alpha=0.2, linewidth=0.3, zorder=0)

    # (c) QQ-like: sorted residuals
    ax = axes[2]
    from scipy import stats
    sorted_res = np.sort(residuals)
    theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_res)))
    ax.scatter(theoretical_q, sorted_res, alpha=0.6, s=15, color=ACCENT_PURPLE,
               edgecolors='black', linewidths=0.3, zorder=3)
    # Reference line
    slope, intercept = np.polyfit(theoretical_q, sorted_res, 1)
    ax.plot(theoretical_q, slope * theoretical_q + intercept, 'r--', linewidth=0.8, zorder=2)
    ax.set_xlabel('Theoretical Quantiles', fontsize=7.5)
    ax.set_ylabel('Sample Quantiles', fontsize=7.5)
    ax.set_title('(c) Q-Q Plot', fontweight='bold', fontsize=7.5)
    ax.grid(True, alpha=0.2, linewidth=0.3, zorder=0)

    fig.suptitle(f'Residual Analysis — Best M3 Config ({modal_name})', fontweight='bold', fontsize=9, y=1.03)
    fig.tight_layout(w_pad=0.8)
    save_fig(fig, 'fig17_residual_analysis')


# ============================================================================
# MAIN
# ============================================================================
def main():
    print('=' * 70)
    print('GENERATING JOURNAL-QUALITY FIGURES')
    print(f'Output directory: {OUTPUT_DIR}/')
    print('=' * 70)

    fig1a_yield_distributions_corn()
    fig1b_yield_distributions_soybean()
    fig2_modality_band_statistics()
    fig3_m3_modality_ablation()
    fig4_m3_scatter_best()
    fig5_m3_scatter_all()
    fig6_yearwise_performance()
    fig7_classical_ml_temporal()
    fig8_classical_vs_terramind()
    fig9_m4_temporal_ablation()
    fig10_m5_normalization()
    fig11_model_summary()
    fig12_weather_distributions()
    fig13_ndvi_temporal()
    fig14_data_splits()
    fig15_m3_modality_heatmap()
    fig16_yield_statistics()
    fig17_residual_analysis()

    print('\n' + '=' * 70)
    print(f'ALL FIGURES SAVED TO: {OUTPUT_DIR}/')
    print('Formats: PDF (vector) + PNG (raster, 600 DPI)')
    print('=' * 70)

    # Print figure index
    print('\nFIGURE INDEX:')
    print('─' * 50)
    print('Fig 1a: Corn yield distributions by year (2017-2024)')
    print('Fig 1b: Soybean yield distributions by year (2017-2024)')
    print('Fig 2:  Modality band statistics')
    print('Fig 3:  M3 modality ablation (R² + MAE bars)')
    print('Fig 4:  M3 best config scatter plots (2×2)')
    print('Fig 5:  M3 all config scatter plots (grid)')
    print('Fig 6:  Year-wise TerraMind performance')
    print('Fig 7:  Classical ML weekly temporal R²')
    print('Fig 8:  Classical ML vs TerraMind comparison')
    print('Fig 9:  M4 temporal ablation (R² vs window)')
    print('Fig 10: M5 normalization experiments')
    print('Fig 11: Model summary M1–M5')
    print('Fig 12: Weather band distributions')
    print('Fig 13: NDVI temporal profiles')
    print('Fig 14: Data split summary')
    print('Fig 15: M3 modality heatmap')
    print('Fig 16: Yield statistics by year')
    print('Fig 17: Residual analysis')


if __name__ == '__main__':
    main()
