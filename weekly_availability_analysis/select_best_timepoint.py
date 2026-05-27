import pandas as pd
import numpy as np
import os
import glob

def select_best_weeks_across_years(csv_dir, min_weeks=12, max_weeks=18, output_path=None):
    """
    Select best weeks based on average quality across all years
    Also considers growing stages to ensure critical periods are covered
    
    Args:
        csv_dir: Directory containing availability CSV files
        min_weeks: Minimum number of weeks to select
        max_weeks: Maximum number of weeks to select
        output_path: Path to save results (optional)
    
    Returns:
        selected_weeks: List of selected week indices (common across all years)
    """
    
    # Load all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, 'week_availability_*.csv'))
    
    if not csv_files:
        csv_files = [csv_dir]
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Filter present data
    df_present = df_all[df_all['Status'] == 'present'].copy()
    
    # Calculate quality score for each week-year combination
    week_year_quality = []
    
    years = df_present['Year'].unique()
    print(f"Found {len(years)} years: {sorted(years)}")
    
    for year in years:
        df_year = df_present[df_present['Year'] == year]
        
        for week in range(24):
            s1_pct = df_year[(df_year['Modality'] == 'S1GRD') & (df_year['Week'] == week)]['Percentage']
            s2_pct = df_year[(df_year['Modality'] == 'S2L2A') & (df_year['Week'] == week)]['Percentage']
            
            s1_pct = s1_pct.values[0] if len(s1_pct) > 0 else 0
            s2_pct = s2_pct.values[0] if len(s2_pct) > 0 else 0
            
            # Quality score: S2 is more important (optical is more important for yield prediction)
            quality = (s1_pct * 0.4) + (s2_pct * 0.6)
            
            week_label = df_year[df_year['Week'] == week]['WeekLabel'].values
            week_label = week_label[0] if len(week_label) > 0 else f"Week {week}"
            
            week_year_quality.append({
                'year': year,
                'week': week,
                'week_label': week_label,
                's1_pct': s1_pct,
                's2_pct': s2_pct,
                'quality': quality
            })
    
    df_quality = pd.DataFrame(week_year_quality)
    
    # ========================================
    # Calculate average quality across all years
    # ========================================
    avg_quality = df_quality.groupby('week').agg({
        's1_pct': 'mean',
        's2_pct': 'mean',
        'quality': 'mean',
        'week_label': 'first'
    }).reset_index()
    
    # Add growing stage info
    def get_growing_stage(week):
        if week <= 3:      # Apr W1-W4
            return 'Planting'
        elif week <= 7:    # May W1-W4
            return 'Emergence'
        elif week <= 11:   # Jun W1-W4
            return 'Vegetative'
        elif week <= 15:   # Jul W1-W4
            return 'Reproductive'  # CRITICAL!
        elif week <= 19:   # Aug W1-W4
            return 'Grain Fill'    # CRITICAL!
        else:              # Sep W1-W4
            return 'Maturity'
    
    avg_quality['stage'] = avg_quality['week'].apply(get_growing_stage)
    avg_quality['is_critical'] = avg_quality['stage'].isin(['Reproductive', 'Grain Fill'])
    
    print("\n" + "="*70)
    print("AVERAGE QUALITY ACROSS ALL YEARS")
    print("="*70)
    print(f"{'Week':<12} {'Stage':<15} {'S1 %':>8} {'S2 %':>8} {'Quality':>10} {'Critical':>10}")
    print("-"*70)
    for _, row in avg_quality.iterrows():
        critical_marker = "★★★" if row['is_critical'] else ""
        print(f"{row['week_label']:<12} {row['stage']:<15} {row['s1_pct']:>8.1f} {row['s2_pct']:>8.1f} {row['quality']:>10.1f} {critical_marker:>10}")
    
    # ========================================
    # Week selection logic
    # ========================================
    selected = []
    
    # 1. Select top 6 weeks from critical period (Jul-Aug, Reproductive & Grain Fill)
    critical_weeks = avg_quality[avg_quality['is_critical']].nlargest(6, 'quality')
    selected.extend(critical_weeks['week'].tolist())
    print(f"\n[Step 1] Selected {len(critical_weeks)} weeks from critical period: {sorted(critical_weeks['week'].tolist())}")
    
    # 2. Select at least 1-2 weeks from each non-critical stage
    for stage in ['Planting', 'Emergence', 'Vegetative', 'Maturity']:
        stage_weeks = avg_quality[(avg_quality['stage'] == stage) & (~avg_quality['week'].isin(selected))]
        if len(stage_weeks) > 0:
            # Select top 2 weeks by quality from each stage
            top_weeks = stage_weeks.nlargest(2, 'quality')
            selected.extend(top_weeks['week'].tolist())
    
    print(f"[Step 2] After adding weeks from each stage: {len(selected)} weeks")
    
    # 3. If minimum count not reached, add more weeks by quality score
    if len(selected) < min_weeks:
        remaining = avg_quality[~avg_quality['week'].isin(selected)]
        remaining = remaining.nlargest(min_weeks - len(selected), 'quality')
        selected.extend(remaining['week'].tolist())
        print(f"[Step 3] After filling minimum count: {len(selected)} weeks")
    
    # 4. If maximum count exceeded, remove lowest quality weeks
    if len(selected) > max_weeks:
        selected_df = avg_quality[avg_quality['week'].isin(selected)]
        selected_df = selected_df.nlargest(max_weeks, 'quality')
        selected = selected_df['week'].tolist()
        print(f"[Step 4] After capping at maximum count: {len(selected)} weeks")
    
    # Sort
    selected = sorted(set(selected))
    
    # ========================================
    # Print results
    # ========================================
    print("\n" + "="*70)
    print(f"FINAL SELECTED WEEKS: {len(selected)}")
    print("="*70)
    print(f"Week indices: {selected}")
    print("\nDetails:")
    print(f"{'Week':<12} {'Stage':<15} {'S1 %':>8} {'S2 %':>8} {'Quality':>10}")
    print("-"*60)
    
    for week in selected:
        row = avg_quality[avg_quality['week'] == week].iloc[0]
        print(f"{row['week_label']:<12} {row['stage']:<15} {row['s1_pct']:>8.1f} {row['s2_pct']:>8.1f} {row['quality']:>10.1f}")
    
    # ========================================
    # Year-by-year validation: check quality of selected weeks
    # ========================================
    print("\n" + "="*70)
    print("YEAR-BY-YEAR QUALITY FOR SELECTED WEEKS")
    print("="*70)
    
    for year in sorted(years):
        df_year = df_quality[df_quality['year'] == year]
        selected_quality = df_year[df_year['week'].isin(selected)]['quality']
        avg_q = selected_quality.mean()
        min_q = selected_quality.min()
        max_q = selected_quality.max()
        print(f"Year {year}: Avg={avg_q:.1f}, Min={min_q:.1f}, Max={max_q:.1f}")
    
    # ========================================
    # Save results
    # ========================================
    if output_path:
        result_df = avg_quality[avg_quality['week'].isin(selected)].copy()
        result_df['selected'] = True
        result_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return selected, avg_quality


def print_week_labels(selected_weeks):
    """Print labels for selected weeks"""
    week_labels = [
        'Apr W1', 'Apr W2', 'Apr W3', 'Apr W4',
        'May W1', 'May W2', 'May W3', 'May W4',
        'Jun W1', 'Jun W2', 'Jun W3', 'Jun W4',
        'Jul W1', 'Jul W2', 'Jul W3', 'Jul W4',
        'Aug W1', 'Aug W2', 'Aug W3', 'Aug W4',
        'Sep W1', 'Sep W2', 'Sep W3', 'Sep W4'
    ]
    
    print("\nSelected week labels:")
    for w in selected_weeks:
        print(f"  Week {w}: {week_labels[w]}")


if __name__ == "__main__":
    csv_path = 'week_availability_stats_all_years.csv'
    
    selected_weeks, quality_df = select_best_weeks_across_years(
        csv_path,
        min_weeks=12,
        max_weeks=16,
        output_path='selected_weeks_result.csv'
    )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Final selected weeks: {selected_weeks}")
    print(f"Total: {len(selected_weeks)} weeks")
    print_week_labels(selected_weeks)
    
    # Print excluded weeks as well
    all_weeks = set(range(24))
    excluded = sorted(all_weeks - set(selected_weeks))
    print(f"\nExcluded weeks: {excluded}")
    print_week_labels(excluded)s