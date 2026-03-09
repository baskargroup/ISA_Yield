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
# JOURNAL-STANDARD FONT SIZES (consistent across all figures)
# ============================================================================
# Uniform font size across all figure elements
FONT_SIZE = 10
SUPTITLE_SIZE = FONT_SIZE
TITLE_SIZE = FONT_SIZE
LABEL_SIZE = FONT_SIZE
TICK_SIZE = FONT_SIZE
LEGEND_SIZE = FONT_SIZE
ANNOT_SIZE = FONT_SIZE
INSET_SIZE = FONT_SIZE
# Dense grid plots (e.g. fig5 with many small subplots)
SMALL_TITLE_SIZE = 6
SMALL_TICK_SIZE = 6
SMALL_LABEL_SIZE = 6

# ============================================================================
# JOURNAL-STANDARD MATPLOTLIB RCPARAMS
# ============================================================================
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'STIX', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': FONT_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'figure.titlesize': SUPTITLE_SIZE,
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
                    ha='center', va='center', fontsize=ANNOT_SIZE, color='gray')
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
            ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=INSET_SIZE,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray', lw=0.4))

    for j in range(len(years), nrows * ncols):
        axes_flat[j].set_visible(False)

    for j in range(nrows):
        axes[j, 0].set_ylabel('No. of Fields')
    for j in range(ncols):
        axes[-1, j].set_xlabel('Yield (bu/acre)')

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
                    ha='center', va='center', fontsize=ANNOT_SIZE, color='gray')
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
            ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=INSET_SIZE,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray', lw=0.4))

    for j in range(len(years), nrows * ncols):
        axes_flat[j].set_visible(False)

    for j in range(nrows):
        axes[j, 0].set_ylabel('No. of Fields')
    for j in range(ncols):
        axes[-1, j].set_xlabel('Yield (bu/acre)')

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
                  'names': ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'],
                  'ylabel': 'Reflectance (×10⁴)'},
        'S1GRD': {'bands': 2, 'label': 'Sentinel-1', 'names': ['VV','VH'],
                  'ylabel': 'Backscatter (dB)'},
        'DEM':   {'bands': 1, 'label': 'DEM', 'names': ['Elevation'],
                  'ylabel': 'Elevation (m)'},
        'WEATHER': {'bands': 7, 'label': 'Weather',
                    'names': ['PRCP (mm/d)','TMAX (°C)','TMIN (°C)','SRAD (W/m²)','VP (Pa)','SWE (kg/m²)','DAYL (s/d)'],
                    'ylabel': 'Value (log scale)'},
        'SOIL':  {'bands': 10, 'label': 'Soil',
                  'names': ['aws100','aws150','aws999','nccpi3all','nccpi3corn',
                            'rootznaws','soc150','soc999','pctearthmc','nccpi3soy'],
                  'ylabel': 'Value (log scale)'},
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

    # Create multi-panel figure: 2 rows (3 top, 2 bottom centred)
    fig = plt.figure(figsize=(DOUBLE_COL, 4.8))
    gs = GridSpec(2, 6, figure=fig, hspace=0.55, wspace=0.65)
    axes = [
        fig.add_subplot(gs[0, 0:2]),  # Sentinel-2
        fig.add_subplot(gs[0, 2:4]),  # Sentinel-1
        fig.add_subplot(gs[0, 4:6]),  # DEM
        fig.add_subplot(gs[1, 1:3]),  # Weather
        fig.add_subplot(gs[1, 3:5]),  # Soil
    ]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

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
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=TICK_SIZE)
        ax.set_title(data['info']['label'], fontweight='bold', fontsize=TITLE_SIZE, pad=3)
        ax.set_ylabel(data['info']['ylabel'], fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)

        # Use log scale for panels where value ranges span orders of magnitude
        if mod_name in ('WEATHER', 'SOIL'):
            ax.set_yscale('log')
            # Ensure all bars are visible: set floor slightly above zero
            ax.set_ylim(bottom=max(1e-3, min(m for m in means if m > 0) * 0.5))

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save_fig(fig, 'fig2_modality_band_statistics')


# ============================================================================
# FIGURE 3: M3 Modality Ablation — R² and MAE comparison (bar chart)
# ============================================================================
def fig3_m3_modality_ablation():
    """M3 ablation study: R² and MAE across modality combinations, per crop."""
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

    # ---- Figure: 2 panels (Corn R², Soybean R²) with MAE annotations ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 4.5))

    # Panel (a): Corn R²
    corn_df = rdf[rdf['Crop'] == 'Corn'].sort_values('R2', ascending=True)
    y_pos = np.arange(len(corn_df))
    ax_a.barh(y_pos, corn_df['R2'], color=CORN_COLOR, edgecolor='black',
              linewidth=0.3, height=0.7, alpha=0.9)
    for i, (_, row) in enumerate(corn_df.iterrows()):
        ax_a.text(row['R2'] + 0.005, i,
                  f"R\u00b2={row['R2']:.3f}  MAE={row['MAE']:.1f}",
                  va='center', fontsize=ANNOT_SIZE)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(corn_df['Modal'], fontsize=TICK_SIZE)
    ax_a.set_xlabel('$R^2$')
    ax_a.set_title('(a) Corn — $R^2$ by Modality', fontweight='bold', fontsize=TITLE_SIZE)
    ax_a.set_xlim(0, corn_df['R2'].max() * 1.35)

    # Panel (b): Soybean R²
    soy_df = rdf[rdf['Crop'] == 'Soybean'].sort_values('R2', ascending=True)
    y_pos = np.arange(len(soy_df))
    ax_b.barh(y_pos, soy_df['R2'], color=SOYBEAN_COLOR, edgecolor='black',
              linewidth=0.3, height=0.7, alpha=0.9)
    for i, (_, row) in enumerate(soy_df.iterrows()):
        ax_b.text(row['R2'] + 0.005, i,
                  f"R\u00b2={row['R2']:.3f}  MAE={row['MAE']:.1f}",
                  va='center', fontsize=ANNOT_SIZE)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(soy_df['Modal'], fontsize=TICK_SIZE)
    ax_b.set_xlabel('$R^2$')
    ax_b.set_title('(b) Soybean — $R^2$ by Modality', fontweight='bold', fontsize=TITLE_SIZE)
    ax_b.set_xlim(0, soy_df['R2'].max() * 1.35)

    fig.tight_layout(w_pad=1.0)
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

    # Find top-2 configs per crop by R²
    from sklearn.metrics import r2_score
    crop_scores = {}  # {crop: [(modal, r2), ...]}
    for (modal, crop), df in modal_data.items():
        y_true = df['YieldGT'].values
        y_pred = df['Prediction'].values
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        crop_scores.setdefault(crop, []).append((modal, r2))
    for crop in crop_scores:
        crop_scores[crop].sort(key=lambda x: -x[1])

    # Best and 2nd-best per crop
    configs_to_plot = []
    for crop in ['Corn', 'Soybean']:
        for modal, _ in crop_scores.get(crop, [])[:2]:
            configs_to_plot.append((modal, crop))

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
               f'MAPE = {metrics["MAPE"]:.1f}%\n'
               f'n = {metrics["N"]}')
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=INSET_SIZE,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9,
                          ec='gray', lw=0.4))

        letter = chr(97 + idx)
        ax.set_title(f'({letter}) {modal} — {crop}', fontweight='bold', fontsize=TITLE_SIZE, pad=3)
        ax.set_xlabel('Observed Yield (bu/acre)', fontsize=LABEL_SIZE)
        ax.set_ylabel('Predicted Yield (bu/acre)', fontsize=LABEL_SIZE)
        ax.legend(loc='lower right', fontsize=LEGEND_SIZE, framealpha=0.9)
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

        ax.set_title(f'{modal} {crop}\n$R^2$={metrics["R2"]:.3f}', fontsize=SMALL_TITLE_SIZE, pad=2)
        ax.tick_params(labelsize=SMALL_TICK_SIZE)

        if idx % ncols == 0:
            ax.set_ylabel('Pred', fontsize=SMALL_LABEL_SIZE)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel('Obs', fontsize=SMALL_LABEL_SIZE)

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
                 f'{val:.3f}', ha='center', va='bottom', fontsize=ANNOT_SIZE, color=NEUTRAL_BLUE,
                 fontweight='bold')
    for bar, val in zip(bars2, df['MAE']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=ANNOT_SIZE, color=ACCENT_RED,
                 fontweight='bold')

    # Combined legend
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc='upper left', fontsize=LEGEND_SIZE, framealpha=0.9)

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
    ax1.set_title('(a) Test $R^2$ by Week', fontweight='bold', fontsize=TITLE_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax1.grid(True, alpha=0.2, linewidth=0.3)
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    # Panel (b): Test MAE
    for model, color, marker in [('XGBoost', NEUTRAL_BLUE, 'o'), ('PLSR', ACCENT_RED, 's')]:
        sub = df[df['model'] == model]
        ax2.plot(sub['week'], sub['test_mae'], color=color, marker=marker,
                 markersize=3, linewidth=0.8, label=model, zorder=3)

    ax2.set_xlabel('Week')
    ax2.set_ylabel('Test MAE')
    ax2.set_title('(b) Test MAE by Week', fontweight='bold', fontsize=TITLE_SIZE)
    ax2.grid(True, alpha=0.2, linewidth=0.3)

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, 'fig7_classical_ml_temporal_weekly')


# ============================================================================
# FIGURE 8: Classical ML vs TerraMind Comparison
# ============================================================================
def fig8_classical_vs_terramind():
    """Compare classical ML baselines with TerraMind deep learning models, per crop."""
    print('Figure 8: Classical ML vs TerraMind comparison...')

    # Classical ML best results (combined — no per-crop split available)
    classical = pd.read_csv('classical_ml_summary_s12wd_biweek12.csv')

    # Compute M3 test metrics per crop from prediction CSVs
    pred_dir = 'M3/predictions'
    m3_crop = {}  # {(modal, crop): {R2, MAE}}
    for csv_file in sorted(Path(pred_dir).glob('*.csv')):
        modal = extract_modal_code(csv_file.stem)
        crop = extract_crop(csv_file.stem)
        df = pd.read_csv(csv_file)
        yt = df['YieldGT'].values
        yp = df['Prediction'].values
        m3_crop[(modal, crop)] = {'R2': r2_score(yt, yp),
                                  'MAE': mean_absolute_error(yt, yp)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    for ax, crop_name, crop_color in [
        (ax1, 'Corn', CORN_COLOR), (ax2, 'Soybean', SOYBEAN_COLOR)
    ]:
        # Classical ML (combined data — shown as reference)
        models = ['PLSR*', 'XGBoost*']
        r2_vals = [classical[classical['model'] == m]['test_r2'].values[0]
                   for m in ['PLSR', 'XGBoost']]

        # Top 3 M3 configs for this crop
        crop_results = {k: v for k, v in m3_crop.items() if k[1] == crop_name}
        sorted_crop = sorted(crop_results.items(), key=lambda x: x[1]['R2'], reverse=True)[:3]
        for (modal, _), met in sorted_crop:
            models.append(f'TM-{modal}')
            r2_vals.append(met['R2'])

        x = np.arange(len(models))
        colors = [ACCENT_RED, NEUTRAL_BLUE] + [crop_color] * len(sorted_crop)

        bars = ax.bar(x, r2_vals, color=colors, edgecolor='black', linewidth=0.4,
                      alpha=0.85, zorder=3)
        for bar, val in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=ANNOT_SIZE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=TICK_SIZE)
        ax.set_ylabel('Test $R^2$')
        letter = '(a)' if crop_name == 'Corn' else '(b)'
        ax.set_title(f'{letter} {crop_name} — $R^2$ Comparison', fontweight='bold', fontsize=TITLE_SIZE)
        ax.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)
        ax.set_ylim(0, max(r2_vals) * 1.2)

    # Note about classical ML
    ax1.text(0.02, 0.02, '* Classical ML trained on\n  combined normalized yield',
             transform=ax1.transAxes, fontsize=INSET_SIZE, style='italic',
             bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='orange', lw=0.3))

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
    ax1.set_title('(a) $R^2$ vs Temporal Window', fontweight='bold', fontsize=TITLE_SIZE)
    ax1.grid(True, alpha=0.2, linewidth=0.3)
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    # Annotate key points
    best_row = tdf_best.loc[tdf_best['val_R2'].idxmax()]
    ax1.annotate(f"Best: wk {int(best_row['week'])}\n$R^2$={best_row['val_R2']:.3f}",
                 xy=(best_row['week'], best_row['val_R2']),
                 xytext=(best_row['week']-2, best_row['val_R2']-0.1),
                 fontsize=ANNOT_SIZE, arrowprops=dict(arrowstyle='->', color='black', lw=0.6),
                 bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='gray', lw=0.4))

    # MAE vs week
    ax2.plot(tdf_best['week'], tdf_best['val_MAE'], color=ACCENT_RED,
             marker='s', markersize=4, linewidth=1.2, zorder=3)
    ax2.fill_between(tdf_best['week'], tdf_best['val_MAE'], alpha=0.15,
                     color=ACCENT_RED)
    ax2.set_xlabel('Observation Window (weeks)')
    ax2.set_ylabel('Validation MAE')
    ax2.set_title('(b) MAE vs Temporal Window', fontweight='bold', fontsize=TITLE_SIZE)
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
        ax.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=ANNOT_SIZE, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_ylabel('Validation $R^2$')
    ax.set_title('M5: Different Normalization Experiments', fontweight='bold', fontsize=TITLE_SIZE)
    ax.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)

    fig.tight_layout()
    save_fig(fig, 'fig10_m5_normalization')


# ============================================================================
# FIGURE 11: Comprehensive Model Comparison (M1–M5 summary)
# ============================================================================
def fig11_model_summary():
    """Summary comparison across all 5 model experiments, per crop."""
    print('Figure 11: Model summary M1–M5...')

    import json

    # ---- Collect per-crop results from each model ----

    # M3: per-crop from prediction CSVs — find best config per crop
    pred_dir = 'M3/predictions'
    m3_crop = {}  # {(modal, crop): (yt, yp)}
    for csv_file in sorted(Path(pred_dir).glob('*.csv')):
        modal = extract_modal_code(csv_file.stem)
        crop = extract_crop(csv_file.stem)
        df = pd.read_csv(csv_file)
        m3_crop[(modal, crop)] = (df['YieldGT'].values, df['Prediction'].values)

    # M1/M2: per-crop from predictions/
    m1_crop, m2_crop = {}, {}
    for csv_file in sorted(Path('predictions').glob('*.csv')):
        modal = extract_modal_code(csv_file.stem)
        crop = extract_crop(csv_file.stem)
        df = pd.read_csv(csv_file)
        yt, yp = df['YieldGT'].values, df['Prediction'].values
        if modal == 'S12d':
            m1_crop[crop] = (yt, yp)
        if modal in ['S12cdws', 'S12cdw']:
            m2_crop[crop] = (yt, yp)

    # M4: from wandb (best weekly temporal — combined validation, no per-crop)
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

    # M5: from wandb (combined validation, no per-crop)
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

    # Build per-crop summary rows
    rows = []
    for crop_name in ['Corn', 'Soybean']:
        # M1
        if crop_name in m1_crop:
            yt, yp = m1_crop[crop_name]
            rows.append({'Model': 'M1', 'Crop': crop_name, 'Config': 'S12d (frozen)',
                         'R2': r2_score(yt, yp), 'MAE': mean_absolute_error(yt, yp)})
        # M2
        if crop_name in m2_crop:
            yt, yp = m2_crop[crop_name]
            rows.append({'Model': 'M2', 'Crop': crop_name, 'Config': 'S12cdw (unfrozen)',
                         'R2': r2_score(yt, yp), 'MAE': mean_absolute_error(yt, yp)})
        # M3 — best config for this crop
        crop_m3 = {k: v for k, v in m3_crop.items() if k[1] == crop_name}
        if crop_m3:
            best_key = max(crop_m3, key=lambda k: r2_score(crop_m3[k][0], crop_m3[k][1]))
            yt, yp = crop_m3[best_key]
            rows.append({'Model': 'M3', 'Crop': crop_name, 'Config': f'{best_key[0]} (ablation)',
                         'R2': r2_score(yt, yp), 'MAE': mean_absolute_error(yt, yp)})

    # M4/M5: combined validation only (no per-crop prediction CSVs)
    rows.append({'Model': 'M4', 'Crop': 'Val', 'Config': 'S12wd (temporal)',
                 'R2': m4_r2, 'MAE': m4_mae})
    rows.append({'Model': 'M5', 'Crop': 'Val', 'Config': 'S12d (diff norm)',
                 'R2': m5_r2, 'MAE': m5_mae})

    summary = pd.DataFrame(rows)

    # ---- Figure: grouped horizontal bar ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, max(2.8, len(summary) * 0.4)),
                                   sharey=True)

    y = np.arange(len(summary))
    colors = []
    for _, row in summary.iterrows():
        if row['Crop'] == 'Corn':
            colors.append(CORN_COLOR)
        elif row['Crop'] == 'Soybean':
            colors.append(SOYBEAN_COLOR)
        else:
            colors.append(MODEL_COLORS.get(row['Model'], '#999999'))

    # R² bars
    bars1 = ax1.barh(y, summary['R2'], color=colors, edgecolor='black',
                     linewidth=0.4, height=0.6, alpha=0.9, zorder=3)
    for i, val in enumerate(summary['R2']):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=ANNOT_SIZE, fontweight='bold')
    ax1.set_yticks(y)
    labels = [f"{row['Model']} {row['Crop']}\n{row['Config']}" for _, row in summary.iterrows()]
    ax1.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax1.set_xlabel('$R^2$ Score')
    ax1.set_title('(a) $R^2$ Comparison', fontweight='bold', fontsize=TITLE_SIZE)
    ax1.grid(axis='x', alpha=0.2, linewidth=0.3, zorder=0)
    ax1.set_xlim(0, summary['R2'].max() * 1.2)

    # MAE bars
    bars2 = ax2.barh(y, summary['MAE'], color=colors, edgecolor='black',
                     linewidth=0.4, height=0.6, alpha=0.9, zorder=3)
    for i, val in enumerate(summary['MAE']):
        label = f'{val:.2f}' if val < 1 else f'{val:.1f}'
        ax2.text(val + max(summary['MAE'])*0.02, i, label, va='center', fontsize=ANNOT_SIZE, fontweight='bold')
    ax2.set_xlabel('MAE')
    ax2.set_title('(b) MAE Comparison', fontweight='bold', fontsize=TITLE_SIZE)
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
    # Actual band order in .npy files (verified by physical ranges)
    band_names = ['PRCP', 'TMAX', 'TMIN', 'SRAD', 'VP', 'SWE', 'DAYL']
    # Daymet units for each band
    band_units = {
        'TMAX': '°C', 'PRCP': 'mm/day', 'TMIN': '°C', 'SRAD': 'W/m²',
        'VP': 'Pa', 'SWE': 'kg/m²', 'DAYL': 's/day',
    }

    # Use ALL files for field-level distributions
    all_files = sorted(os.listdir(base))

    band_data = {name: [] for name in band_names}

    for fname in all_files:
        fpath = os.path.join(base, fname)
        arr = np.load(fpath, mmap_mode='r')  # (24, 7, 224, 224)
        # Weather is broadcast: all pixels identical. Read one pixel across weeks.
        pixel_ts = np.array(arr[:, :, 0, 0])  # (24, 7) — force read into memory
        for b, name in enumerate(band_names):
            if b < pixel_ts.shape[1]:
                week_vals = pixel_ts[:, b]
                valid = week_vals[~np.isnan(week_vals)]
                if len(valid) > 0:
                    band_data[name].append(float(np.mean(valid)))

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL, 3.8))
    axes = axes.flatten()
    colors = sns.color_palette('Set2', len(band_names))

    for idx, (name, vals) in enumerate(band_data.items()):
        ax = axes[idx]
        vals = np.array(vals)
        # Use tighter percentile clipping for skewed bands
        if name == 'SWE':
            lo_pct, hi_pct = 0, 80  # heavily zero-inflated
        else:
            lo_pct, hi_pct = 2, 98
        p_lo, p_hi = np.percentile(vals, [lo_pct, hi_pct])
        vals_clipped = vals[(vals >= p_lo) & (vals <= p_hi)]
        span = p_hi - p_lo
        margin = span * 0.03
        n_bins = 25 if name == 'SWE' else 50
        ax.hist(vals_clipped, bins=n_bins, color=colors[idx], alpha=0.8, edgecolor='white',
                linewidth=0.2, density=False, range=(p_lo, p_hi))
        ax.set_xlim(p_lo - margin, p_hi + margin)
        # SWE: log y-scale because the zero-bin dominates (~45% of fields)
        if name == 'SWE':
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.8)  # avoid log(0)
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.set_yticks([1, 10, 100])
        unit = band_units[name]
        ax.set_title(f'{name} ({unit})', fontweight='bold', fontsize=TITLE_SIZE, pad=6)
        ax.tick_params(labelsize=TICK_SIZE)
        # Clean up tick formatting
        if name != 'SWE':
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
        # Per-band x-axis formatting to avoid overlapping / duplicate labels
        if name == 'VP':
            # Range ~1300-1700: use plain integers with 4 ticks
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        elif name == 'DAYL':
            # Range ~49600-50200: show ×10³ s with 4 ticks
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.1f}'))
            ax.set_xlabel('×10³', fontsize=TICK_SIZE, labelpad=1)
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
            ax.ticklabel_format(axis='x', style='plain')
        # Use K notation for y-axis counts if large
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda y, _: f'{y/1000:.0f}K' if y >= 1000 else f'{int(y)}'))
        mu = vals_clipped.mean()
        sigma = vals_clipped.std()
        if mu >= 1000:
            txt = f'$\\mu$={mu/1000:.1f}K\n$\\sigma$={sigma/1000:.1f}K'
        else:
            txt = f'$\\mu$={mu:.1f}\n$\\sigma$={sigma:.1f}'
        ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=INSET_SIZE,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray', lw=0.3))

    # Hide last unused axis
    axes[7].set_visible(False)

    fig.tight_layout(h_pad=1.0, w_pad=0.5)
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

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))

    # Mean ± std
    for data, label, color in [(corn_ndvi, 'Corn', CORN_COLOR), (soy_ndvi, 'Soybean', SOYBEAN_COLOR)]:
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        ax.plot(weeks, mean, color=color, linewidth=1.2, label=label, zorder=3)
        ax.fill_between(weeks, mean - std, mean + std, alpha=0.2, color=color, zorder=2)

    ax.set_xlabel('Week', fontsize=LABEL_SIZE)
    ax.set_ylabel('Mean NDVI', fontsize=LABEL_SIZE)
    ax.set_title('NDVI Temporal Profile', fontweight='bold', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
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
        ax1.text(i - w/2, c + 1, str(c), ha='center', va='bottom', fontsize=ANNOT_SIZE, fontweight='bold')
        ax1.text(i + w/2, s + 1, str(s), ha='center', va='bottom', fontsize=ANNOT_SIZE, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(splits_ordered)
    ax1.set_ylabel('Number of Fields')
    ax1.set_title('(a) Train/Val/Test Split', fontweight='bold', fontsize=TITLE_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
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
    ax2.set_xticklabels(years, rotation=45, ha='right', fontsize=TICK_SIZE)
    ax2.set_ylabel('Number of Fields')
    ax2.set_title('(b) Training Data by Year', fontweight='bold', fontsize=TITLE_SIZE)
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

    # Build matrix (per-crop only)
    modalities = sorted(set(m for m, _ in results.keys()))
    crops = ['Corn', 'Soybean']

    matrix = np.full((len(modalities), len(crops)), np.nan)
    for i, mod in enumerate(modalities):
        for j, crop in enumerate(crops):
            if (mod, crop) in results:
                matrix[i, j] = results[(mod, crop)]

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, max(2.5, len(modalities) * 0.35)))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(crops)))
    ax.set_xticklabels(crops, fontsize=TICK_SIZE)
    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels(modalities, fontsize=TICK_SIZE)

    # Annotate cells
    for i in range(len(modalities)):
        for j in range(len(crops)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.3 or val > 0.85 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=ANNOT_SIZE, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$R^2$ Score', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    ax.set_title('M3: Modality Ablation $R^2$ Heatmap', fontweight='bold', fontsize=TITLE_SIZE, pad=8)

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
    ax1.set_xticklabels(sdf['Year'].astype(str), rotation=45, ha='right', fontsize=TICK_SIZE)
    ax1.set_ylabel('Mean Field Yield (bu/acre)')
    ax1.set_title('(a) Mean Field Yield by Year and Crop', fontweight='bold', fontsize=TITLE_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.2, linewidth=0.3, zorder=0)

    # Panel (b): Number of fields
    ax2.bar(x - w/2, sdf['n_corn'], w, color=CORN_COLOR, edgecolor='black',
            linewidth=0.4, label='Corn', alpha=0.85)
    ax2.bar(x + w/2, sdf['n_soy'], w, color=SOYBEAN_COLOR, edgecolor='black',
            linewidth=0.4, label='Soybean', alpha=0.85)
    for i, (c, s) in enumerate(zip(sdf['n_corn'], sdf['n_soy'])):
        ax2.text(i - w/2, c + 0.5, str(c), ha='center', va='bottom', fontsize=ANNOT_SIZE, fontweight='bold')
        ax2.text(i + w/2, s + 0.5, str(s), ha='center', va='bottom', fontsize=ANNOT_SIZE, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sdf['Year'].astype(str), rotation=45, ha='right', fontsize=TICK_SIZE)
    ax2.set_ylabel('Number of Fields')
    ax2.set_title('(b) Number of Fields by Year', fontweight='bold', fontsize=TITLE_SIZE)
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
    ax.set_xlabel('Predicted (bu/acre)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Residual (bu/acre)', fontsize=LABEL_SIZE)
    ax.set_title(f'(a) Residual vs Predicted', fontweight='bold', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.3, zorder=0)

    # (b) Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=30, color=NEUTRAL_BLUE, alpha=0.8, edgecolor='white',
            linewidth=0.3, density=True)
    ax.axvline(x=0, color='black', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Residual (bu/acre)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Density', fontsize=LABEL_SIZE)
    ax.set_title('(b) Residual Distribution', fontweight='bold', fontsize=TITLE_SIZE)
    txt = f'$\\mu$={np.mean(residuals):.1f}\n$\\sigma$={np.std(residuals):.1f}'
    ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=INSET_SIZE,
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
    ax.set_xlabel('Theoretical Quantiles', fontsize=LABEL_SIZE)
    ax.set_ylabel('Sample Quantiles', fontsize=LABEL_SIZE)
    ax.set_title('(c) Q-Q Plot', fontweight='bold', fontsize=TITLE_SIZE)
    ax.grid(True, alpha=0.2, linewidth=0.3, zorder=0)


    fig.tight_layout(w_pad=0.8)
    save_fig(fig, 'fig17_residual_analysis')


# ============================================================================
# FIGURE 18: Yield map for one corn and one soybean field
# ============================================================================
def fig18_yield_maps():
    """Spatial yield maps for one representative corn and one soybean field."""
    print('Figure 18: Yield maps (corn & soybean)...')

    yield_dir = 'processed_data/weekly_24/processed_data_weekly_24/yield_geotiffs'

    # Pick fields with high valid-pixel coverage for visual clarity
    def pick_best_field(crop_keyword, n_candidates=30):
        files = sorted([f for f in os.listdir(yield_dir) if crop_keyword in f])
        best_file, best_coverage = None, 0
        for f in files[:n_candidates]:
            arr = np.load(os.path.join(yield_dir, f))[0]
            valid = np.sum((~np.isnan(arr)) & (arr > 0))
            if valid > best_coverage:
                best_coverage = valid
                best_file = f
        return best_file

    corn_file = pick_best_field('Corn')
    soy_file = pick_best_field('Soybean')

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    # Crop-specific denormalization: normalized = (yield - data_min) / (data_max - data_min)
    denorm = {
        'Corn':    {'data_min': 50.0, 'data_max': 370.0},   # range 320
        'Soybean': {'data_min': 30.0, 'data_max': 120.0},   # range 90
    }

    for ax, fname, crop, cmap_color in [
        (axes[0], corn_file, 'Corn', 'YlOrBr'),
        (axes[1], soy_file, 'Soybean', 'YlGn'),
    ]:
        arr = np.load(os.path.join(yield_dir, fname))[0]  # (224, 224)

        # Convert normalized yield back to bu/acre
        d = denorm[crop]
        arr_bu = arr * (d['data_max'] - d['data_min']) + d['data_min']

        # Mask invalid pixels (originally <=0 in normalized space)
        masked = np.ma.masked_where(np.isnan(arr) | (arr <= 0), arr_bu)

        valid_bu = arr_bu[(~np.isnan(arr)) & (arr > 0)]
        im = ax.imshow(masked, cmap=cmap_color, interpolation='nearest',
                       vmin=np.nanpercentile(valid_bu, 2),
                       vmax=np.nanpercentile(valid_bu, 98))
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Yield (bu/acre)', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

        field_id = fname.replace('.npy', '')
        ax.set_title(f'{crop} — {field_id}', fontweight='bold', fontsize=TITLE_SIZE, pad=4)
        ax.set_xlabel('Pixel Column', fontsize=LABEL_SIZE)
        ax.set_ylabel('Pixel Row', fontsize=LABEL_SIZE)

        # Stats inset (in bu/acre)
        txt = f'$\mu$={valid_bu.mean():.1f} bu/ac\n$\sigma$={valid_bu.std():.1f} bu/ac\nn={len(valid_bu):,} px'
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=INSET_SIZE,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85,
                          ec='gray', lw=0.4))

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, 'fig18_yield_maps')


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
    fig18_yield_maps()

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
    print('Fig 18: Yield maps (corn & soybean)')


if __name__ == '__main__':
    main()
