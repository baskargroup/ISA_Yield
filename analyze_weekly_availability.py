import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import defaultdict
import json
import argparse

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

def parse_used_dates_log(log_path):
    """
    Parse dates_log.txt and return a dict: {filename: {modality: [date1, date2, ...]}}
    
    Expected format:
    ======================================================================
    ST2017IA0018_Soybean.tif (Field: ST2017IA0018, Year: 2017)
      S2L2A: 10 dates
        2017-05-06, 2017-06-03, ...
      S1GRD: 15 dates
        2017-04-01, 2017-04-15, ...
      WEATHER: 27 dates
        2017-04-01, 2017-04-08, ...
    """
    used_dates_dict = {}
    current_file = None
    current_modality = None
    
    with open(log_path, "r") as f:
        for line in f:
            line = line.rstrip()
            
            # Skip separator lines
            if line.startswith('='):
                continue
            
            # Check for filename line (e.g., "ST2017IA0018_Soybean.tif (Field: ...")
            if line and not line.startswith(' ') and '.tif' in line:
                # Extract filename (everything before the space and parenthesis)
                current_file = line.split(' ')[0].strip()
                used_dates_dict[current_file] = {}
                current_modality = None
                continue
            
            # Check for modality line (e.g., "  S2L2A: 10 dates")
            if current_file and ':' in line and 'dates' in line:
                parts = line.strip().split(':')
                current_modality = parts[0].strip()
                used_dates_dict[current_file][current_modality] = []
                continue
            
            # Check for dates line (e.g., "    2017-05-06, 2017-06-03, ...")
            if current_file and current_modality and line.strip():
                dates = [d.strip() for d in line.strip().split(',') if d.strip()]
                used_dates_dict[current_file][current_modality].extend(dates)
    
    return used_dates_dict

def get_weekly_indices(dates):
    """
    Group dates by week (April 1 to Sept 30, 4 weeks per month).
    Returns a list of lists, each containing indices of dates for each week.
    Total: 24 weeks (6 months * 4 weeks)
    """
    week_to_indices = defaultdict(list)
    for idx, d in enumerate(dates):
        m = int(d[5:7])
        day = int(d[8:10])
        if 4 <= m <= 9:
            # week 0-3: April W1-W4, week 4-7: May W1-W4, ..., week 20-23: Sep W1-W4
            week_in_month = min((day - 1) // 7, 3)  # 0, 1, 2, or 3 (cap at 3 for days 22-30/31)
            week = (m - 4) * 4 + week_in_month
            week_to_indices[week].append(idx)
    
    weekly_indices = []
    for w in range(24):  # 6 months * 4 weeks
        weekly_indices.append(week_to_indices[w])
    return weekly_indices

def analyze_week_availability(file, data, weekly_indices):
    """
    Analyze availability status for each week.
    Returns a list of 24 strings: 'present', 'imputed', or 'absent' for each week.
    
    Logic:
    - 'present': week has actual data (non-zero slices exist in the raw data)
    - 'imputed': week had no data but was filled using interpolation (had surrounding data)
    - 'absent': week was filled with zeros (no data and no surrounding data to interpolate)
    """
    T, C, H, W = data.shape
    availability_status = []
    
    # First pass: Determine which weeks have actual data
    has_actual_data = [False] * 24
    for i, inds in enumerate(weekly_indices):
        if inds:
            # Check if any of the data slices are non-zero
            has_non_zero = False
            for idx in inds:
                if not np.all(data[idx] == 0):
                    has_non_zero = True
                    break
            has_actual_data[i] = has_non_zero
    
    # Second pass: Determine status for each week
    for i in range(24):
        if has_actual_data[i]:
            availability_status.append('present')
        else:
            # Check if there's surrounding data for interpolation
            has_prev = False
            has_next = False
            
            # Check previous weeks
            for j in range(i-1, -1, -1):
                if has_actual_data[j]:
                    has_prev = True
                    break
            
            # Check next weeks
            for j in range(i+1, 24):
                if has_actual_data[j]:
                    has_next = True
                    break
            
            # If has surrounding data, it would be imputed; otherwise absent
            if has_prev or has_next:
                availability_status.append('imputed')
            else:
                availability_status.append('absent')
    
    return availability_status

def analyze_processed_week_availability(data):
    """
    Analyze availability status for processed data (after aggregation).
    This checks the final processed data to determine which weeks are:
    - 'present': Contains non-zero data
    - 'absent': All zeros
    """
    T, C, H, W = data.shape
    availability_status = []
    
    for t in range(T):
        # Check if the entire time slice is zero
        if np.all(data[t] == 0):
            availability_status.append('absent')
        else:
            # Check if it looks like it was interpolated (hard to distinguish from original)
            # For now, we'll mark as present if non-zero
            availability_status.append('present')
    
    return availability_status

def extract_year_from_filename(filename):
    """
    Extract year from filename (e.g., ST2019IA0050_Soybean.npy -> 2019)
    """
    import re
    match = re.search(r'(\d{4})', filename)
    if match:
        return match.group(1)
    return 'unknown'

def analyze_modality(modality_path, modality_name, used_dates_dict, is_processed=False):
    """
    Analyze week availability for all files in a modality folder.
    Returns a dictionary with file-level availability statistics.
    """
    # Look for both .npy and .tif.npy files
    files = sorted(glob.glob(os.path.join(modality_path, '*.npy')))
    
    results = {}
    
    for file in files:
        try:
            data = np.load(file)
            
            # Get base filename for matching with dates
            base = os.path.basename(file)
            # Try to match with .tif extension for used_dates_dict lookup
            base_for_lookup = base.replace('.npy', '.tif')
            
            # Extract year from filename
            year = extract_year_from_filename(base)
            
            # Get dates for this specific modality
            file_dates_dict = used_dates_dict.get(base_for_lookup, {})
            dates = file_dates_dict.get(modality_name, [])
            
            if is_processed:
                # For processed data, check which weeks are present
                if len(data.shape) == 4 and data.shape[0] >= 24:
                    availability = analyze_processed_week_availability(data[:24])
                else:
                    availability = ['unknown'] * 24
            else:
                # For unprocessed data, analyze based on raw data
                weekly_indices = get_weekly_indices(dates)
                availability = analyze_week_availability(file, data, weekly_indices)
            
            results[os.path.basename(file)] = {
                'availability': availability,
                'dates': dates if dates else [],
                'year': year
            }
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    return results

def create_availability_summary(results_dict):
    """
    Create a summary dataframe with week availability statistics.
    """
    data = []
    
    month_names = ['', '', '', '', 'April', 'May', 'June', 'July', 'August', 'September']
    
    for modality, file_results in results_dict.items():
        for filename, info in file_results.items():
            availability = info['availability']
            year = info.get('year', 'unknown')
            for week_idx, status in enumerate(availability):
                # Determine month and week within month
                month = 4 + week_idx // 4  # April=4, May=5, ..., September=9
                week_in_month = (week_idx % 4) + 1  # 1, 2, 3, or 4
                
                data.append({
                    'Modality': modality,
                    'Filename': filename,
                    'Year': year,
                    'Week': week_idx,
                    'Month': month_names[month],
                    'WeekInMonth': week_in_month,
                    'WeekLabel': f"{month_names[month][:3]} W{week_in_month}",
                    'Status': status
                })
    
    return pd.DataFrame(data)

def plot_availability_heatmap(df, output_path='week_availability_heatmap.png', year=None):
    """
    Create a heatmap showing availability status across weeks for each modality.
    Journal-quality figure optimized for 2-column layout (double column width for 24 weeks).
    """
    # Filter by year if specified
    if year is not None:
        df = df[df['Year'] == str(year)]
        title_suffix = f' ({year})'
    else:
        title_suffix = ' (All Years)'
    
    # Calculate percentage for each status by modality and week
    status_order = ['present', 'imputed', 'absent']
    status_labels = ['Present', 'Imputed', 'Absent']
    
    # Compact week labels for 24 weeks
    week_labels = []
    month_abbrevs = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    for m in month_abbrevs:
        for w in range(1, 5):
            week_labels.append(f'{m}-W{w}')
    
    modalities = sorted(df['Modality'].unique())
    
    # Double column width for 24 weeks
    fig_width = DOUBLE_COLUMN_WIDTH
    fig_height = 2.5 + 0.35 * len(modalities)  # Compact height
    
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height))
    
    # Color schemes optimized for publication (colorblind-friendly)
    cmaps = {
        'present': 'Greens',   # Green for present data
        'imputed': 'Oranges',  # Orange for imputed data
        'absent': 'Reds'       # Red for absent data
    }
    
    for idx, (status, status_label) in enumerate(zip(status_order, status_labels)):
        ax = axes[idx]
        
        # Calculate percentage of files with this status for each modality/week
        pivot_data = []
        for modality in modalities:
            row = []
            for week in range(24):
                subset = df[(df['Modality'] == modality) & (df['Week'] == week)]
                total = len(subset)
                count = len(subset[subset['Status'] == status])
                percentage = (count / total * 100) if total > 0 else 0
                row.append(percentage)
            pivot_data.append(row)
        
        # Create heatmap
        im = ax.imshow(pivot_data, cmap=cmaps[status], 
                      aspect='auto', vmin=0, vmax=100)
        
        # Set labels with proper formatting
        ax.set_xticks(range(24))
        if idx == 2:  # Only show x-labels on bottom plot
            ax.set_xticklabels(week_labels, rotation=45, ha='right', fontsize=6)
            ax.set_xlabel('Week', labelpad=2)
        else:
            ax.set_xticklabels([])
        
        ax.set_yticks(range(len(modalities)))
        ax.set_yticklabels(modalities, fontsize=7)
        ax.set_ylabel(status_label, fontsize=8, fontweight='normal', labelpad=2)
        
        # Add text annotations (only for significant values to reduce clutter)
        for i in range(len(modalities)):
            for j in range(24):
                value = pivot_data[i][j]
                # Only show text if value is significant
                if value > 10:
                    text = ax.text(j, i, f'{value:.0f}',
                                 ha="center", va="center", 
                                 color="white" if value > 50 else "black", 
                                 fontsize=4, fontweight='normal')
        
        # Add colorbar with proper sizing
        cbar = plt.colorbar(im, ax=ax, fraction=0.08, pad=0.02, aspect=10)
        cbar.set_label('%', rotation=0, labelpad=8, fontsize=7, ha='left')
        cbar.ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    
    # Save as both PNG and PDF for publication
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    
    print(f"Saved heatmap to {output_path} and {pdf_path}")
    plt.close()

def plot_availability_stacked_bar(df, output_path='week_availability_stacked.png', year=None):
    """
    Create a stacked bar chart showing availability status distribution across weeks.
    Journal-quality figure optimized for 2-column layout (double column width for 24 weeks).
    """
    # Filter by year if specified
    if year is not None:
        df = df[df['Year'] == str(year)]
        title_suffix = f' ({year})'
    else:
        title_suffix = ' (All Years)'
    
    # Compact week labels for 24 weeks
    week_labels = []
    month_abbrevs = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    for m in month_abbrevs:
        for w in range(1, 5):
            week_labels.append(f'{m}-W{w}')
    
    modalities = sorted(df['Modality'].unique())
    
    # Double column width for 24 weeks
    fig_width = DOUBLE_COLUMN_WIDTH
    fig_height = 1.2 * len(modalities) + 0.5  # Compact height
    
    fig, axes = plt.subplots(len(modalities), 1, figsize=(fig_width, fig_height))
    if len(modalities) == 1:
        axes = [axes]
    
    # Professional, colorblind-friendly colors
    colors = {
        'present': '#2E7D32',   # Dark green
        'imputed': '#F57C00',   # Dark orange  
        'absent': '#C62828'     # Dark red
    }
    
    for idx, modality in enumerate(modalities):
        ax = axes[idx]
        modality_df = df[df['Modality'] == modality]
        
        # Calculate counts for each status per week
        present_counts = []
        imputed_counts = []
        absent_counts = []
        
        for week in range(24):
            subset = modality_df[modality_df['Week'] == week]
            total = len(subset)
            present_counts.append(len(subset[subset['Status'] == 'present']))
            imputed_counts.append(len(subset[subset['Status'] == 'imputed']))
            absent_counts.append(len(subset[subset['Status'] == 'absent']))
        
        x = np.arange(24)
        width = 0.7  # Narrower bars for compact layout
        
        # Create stacked bars
        p1 = ax.bar(x, present_counts, width, label='Present', 
                   color=colors['present'], edgecolor='white', linewidth=0.3)
        p2 = ax.bar(x, imputed_counts, width, bottom=present_counts, 
                   label='Imputed', color=colors['imputed'], 
                   edgecolor='white', linewidth=0.3)
        p3 = ax.bar(x, absent_counts, width, 
                   bottom=np.array(present_counts) + np.array(imputed_counts),
                   label='Absent', color=colors['absent'], 
                   edgecolor='white', linewidth=0.3)
        
        # Formatting
        if idx == len(modalities) - 1:  # Only on last subplot
            ax.set_xlabel('Week', fontsize=8, labelpad=2)
            ax.set_xticklabels(week_labels, fontsize=6, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
        
        ax.set_xticks(x)
        ax.set_ylabel('Count', fontsize=8, labelpad=2)
        ax.set_title(f'{modality}', fontweight='normal', pad=3, fontsize=8, loc='left')
        
        # Grid for readability
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adjust tick parameters
        ax.tick_params(axis='both', which='major', labelsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               frameon=True, framealpha=0.9, edgecolor='gray', fontsize=7,
               bbox_to_anchor=(0.63, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save as both PNG and PDF
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    
    print(f"Saved stacked bar chart to {output_path} and {pdf_path}")
    plt.close()

def save_detailed_statistics(df, output_path='week_availability_stats.csv', year=None):
    """
    Save detailed statistics to CSV file.
    """
    if df.empty:
        print("Warning: No data to save statistics")
        return None
    
    # Filter by year if specified
    if year is not None:
        df = df[df['Year'] == str(year)]
        
    # Overall statistics by modality, year, and week
    stats = df.groupby(['Modality', 'Year', 'Week', 'WeekLabel', 'Status']).size().reset_index(name='Count')
    
    # Calculate percentages
    totals = df.groupby(['Modality', 'Year', 'Week']).size().reset_index(name='Total')
    stats = stats.merge(totals, on=['Modality', 'Year', 'Week'])
    stats['Percentage'] = (stats['Count'] / stats['Total'] * 100).round(2)
    
    stats.to_csv(output_path, index=False)
    print(f"Saved detailed statistics to {output_path}")
    
    return stats

def save_results_json(results_dict, output_path='week_availability_results.json'):
    """
    Save the complete results dictionary to JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved complete results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze weekly availability for S1GRD, S2L2A, and WEATHER')
    parser.add_argument('--data-dir', type=str, default='./chloe_dataset',
                       help='Path to unprocessed data directory')
    parser.add_argument('--log-path', type=str, default='./chloe_dataset/dates_log.txt',
                       help='Path to used_dates_log.txt file')
    parser.add_argument('--output-dir', type=str, default='./weekly_availability_analysis',
                       help='Directory to save output files')
    parser.add_argument('--modalities', nargs='+', default=['S1GRD', 'S2L2A', 'WEATHER'],
                       help='List of modalities to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse used dates log
    print("Parsing used dates log...")
    used_dates_dict = parse_used_dates_log(args.log_path)
    
    # Analyze each modality
    results_dict = {}
    
    for modality in args.modalities:
        modality_path = os.path.join(args.data_dir, modality)
        if not os.path.exists(modality_path):
            print(f"Warning: {modality_path} not found, skipping...")
            continue
        
        print(f"\nAnalyzing {modality}...")
        results = analyze_modality(modality_path, modality, used_dates_dict)
        results_dict[modality] = results
        print(f"Processed {len(results)} files for {modality}")
    
    if not results_dict:
        print("No data found to analyze!")
        return
    
    # Create summary dataframe
    print("\nCreating summary dataframe...")
    df = create_availability_summary(results_dict)
    
    if df.empty:
        print("No availability data to analyze!")
        return
    
    # Save results
    print("\nSaving results...")
    
    # Save JSON
    save_results_json(results_dict, 
                     os.path.join(args.output_dir, 'week_availability_results.json'))
    
    # Get all years in the dataset
    years = sorted(df['Year'].unique())
    print(f"\nFound years: {', '.join(years)}")
    
    # Save overall statistics (all years combined)
    stats_df = save_detailed_statistics(df, 
                                       os.path.join(args.output_dir, 'week_availability_stats_all_years.csv'))
    
    # Generate overall plots (all years combined)
    print("\nGenerating overall plots (all years)...")
    plot_availability_heatmap(df, 
                             os.path.join(args.output_dir, 'week_availability_heatmap_all_years.png'))
    plot_availability_stacked_bar(df, 
                                 os.path.join(args.output_dir, 'week_availability_stacked_all_years.png'))
    
    # Generate year-specific statistics and plots
    for year in years:
        if year == 'unknown':
            continue
        print(f"\nGenerating plots and stats for year {year}...")
        
        # Save year-specific statistics
        save_detailed_statistics(df, 
                               os.path.join(args.output_dir, f'week_availability_stats_{year}.csv'),
                               year=year)
        
        # Generate year-specific plots
        plot_availability_heatmap(df, 
                                 os.path.join(args.output_dir, f'week_availability_heatmap_{year}.png'),
                                 year=year)
        plot_availability_stacked_bar(df, 
                                     os.path.join(args.output_dir, f'week_availability_stacked_{year}.png'),
                                     year=year)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall summary
    print("\n*** OVERALL (All Years) ***")
    for modality in sorted(results_dict.keys()):
        print(f"\n{modality}:")
        modality_df = df[df['Modality'] == modality]
        
        for status in ['present', 'imputed', 'absent']:
            count = len(modality_df[modality_df['Status'] == status])
            total = len(modality_df)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {status.capitalize()}: {count}/{total} ({pct:.1f}%)")
    
    # Year-by-year summary
    for year in years:
        if year == 'unknown':
            continue
        print(f"\n*** YEAR {year} ***")
        year_df = df[df['Year'] == year]
        
        for modality in sorted(results_dict.keys()):
            print(f"\n{modality}:")
            modality_df = year_df[year_df['Modality'] == modality]
            
            if len(modality_df) == 0:
                print("  No data for this year")
                continue
            
            for status in ['present', 'imputed', 'absent']:
                count = len(modality_df[modality_df['Status'] == status])
                total = len(modality_df)
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {status.capitalize()}: {count}/{total} ({pct:.1f}%)")
    
    print(f"\nAll results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()