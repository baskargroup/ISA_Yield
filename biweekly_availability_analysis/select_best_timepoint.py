import pandas as pd
import numpy as np
import os
import glob

def select_best_biweeks_across_years(csv_path, min_biweeks=6, max_biweeks=10, output_path=None):
    """
    모든 년도의 평균 quality를 기반으로 best biweeks 선택
    Growing stage도 고려하여 critical period 보장
    
    Args:
        csv_path: biweek_availability_stats_all_years.csv 파일 경로
        min_biweeks: 최소 선택할 biweek 수 (default: 6)
        max_biweeks: 최대 선택할 biweek 수 (default: 10)
        output_path: 결과 저장 경로 (optional)
    
    Returns:
        selected_biweeks: 선택된 biweek indices 리스트 (모든 년도 공통)
    """
    
    # CSV 파일 로드
    df_all = pd.read_csv(csv_path)
    
    # Present만 필터링
    df_present = df_all[df_all['Status'] == 'present'].copy()
    
    # Biweek별, Year별 quality 계산
    biweek_year_quality = []
    
    years = df_present['Year'].unique()
    print(f"Found {len(years)} years: {sorted(years)}")
    
    for year in years:
        df_year = df_present[df_present['Year'] == year]
        
        for biweek in range(12):
            s1_pct = df_year[(df_year['Modality'] == 'S1GRD') & (df_year['Biweek'] == biweek)]['Percentage']
            s2_pct = df_year[(df_year['Modality'] == 'S2L2A') & (df_year['Biweek'] == biweek)]['Percentage']
            
            s1_pct = s1_pct.values[0] if len(s1_pct) > 0 else 0
            s2_pct = s2_pct.values[0] if len(s2_pct) > 0 else 0
            
            # Quality score: S2 더 중요 (optical이 yield prediction에 더 중요)
            quality = (s1_pct * 0.4) + (s2_pct * 0.6)
            
            biweek_label = df_year[df_year['Biweek'] == biweek]['BiweekLabel'].values
            biweek_label = biweek_label[0] if len(biweek_label) > 0 else f"Biweek {biweek}"
            
            biweek_year_quality.append({
                'year': year,
                'biweek': biweek,
                'biweek_label': biweek_label,
                's1_pct': s1_pct,
                's2_pct': s2_pct,
                'quality': quality
            })
    
    df_quality = pd.DataFrame(biweek_year_quality)
    
    # ========================================
    # 전체 년도 평균 quality 계산
    # ========================================
    avg_quality = df_quality.groupby('biweek').agg({
        's1_pct': 'mean',
        's2_pct': 'mean',
        'quality': 'mean',
        'biweek_label': 'first'
    }).reset_index()
    
    # Growing stage 정보 추가 (Biweekly 기준)
    def get_growing_stage(biweek):
        """
        Biweek 0-1: April (Planting)
        Biweek 2-3: May (Emergence)
        Biweek 4-5: June (Vegetative)
        Biweek 6-7: July (Reproductive) - CRITICAL!
        Biweek 8-9: August (Grain Fill) - CRITICAL!
        Biweek 10-11: September (Maturity)
        """
        if biweek <= 1:      # Apr
            return 'Planting'
        elif biweek <= 3:    # May
            return 'Emergence'
        elif biweek <= 5:    # Jun
            return 'Vegetative'
        elif biweek <= 7:    # Jul
            return 'Reproductive'  # CRITICAL!
        elif biweek <= 9:    # Aug
            return 'Grain Fill'    # CRITICAL!
        else:                # Sep
            return 'Maturity'
    
    avg_quality['stage'] = avg_quality['biweek'].apply(get_growing_stage)
    avg_quality['is_critical'] = avg_quality['stage'].isin(['Reproductive', 'Grain Fill'])
    
    print("\n" + "="*70)
    print("AVERAGE QUALITY ACROSS ALL YEARS (BIWEEKLY)")
    print("="*70)
    print(f"{'Biweek':<15} {'Stage':<15} {'S1 %':>8} {'S2 %':>8} {'Quality':>10} {'Critical':>10}")
    print("-"*70)
    for _, row in avg_quality.iterrows():
        critical_marker = "★★★" if row['is_critical'] else ""
        print(f"{row['biweek_label']:<15} {row['stage']:<15} {row['s1_pct']:>8.1f} {row['s2_pct']:>8.1f} {row['quality']:>10.1f} {critical_marker:>10}")
    
    # ========================================
    # Biweek 선택 로직
    # ========================================
    selected = []
    
    # 1. Critical period (Jul-Aug, Reproductive & Grain Fill)에서 모두 선택 (4개)
    critical_biweeks = avg_quality[avg_quality['is_critical']].nlargest(4, 'quality')
    selected.extend(critical_biweeks['biweek'].tolist())
    print(f"\n[Step 1] Critical period에서 {len(critical_biweeks)}개 선택: {sorted(critical_biweeks['biweek'].tolist())}")
    
    # 2. 각 non-critical stage에서 최소 1개씩 선택 (quality 높은 것)
    for stage in ['Planting', 'Emergence', 'Vegetative', 'Maturity']:
        stage_biweeks = avg_quality[(avg_quality['stage'] == stage) & (~avg_quality['biweek'].isin(selected))]
        if len(stage_biweeks) > 0:
            # 각 stage에서 quality 상위 1개 선택
            top_biweek = stage_biweeks.nlargest(1, 'quality')
            selected.extend(top_biweek['biweek'].tolist())
    
    print(f"[Step 2] 각 stage에서 추가 선택 후: {len(selected)}개")
    
    # 3. 최소 개수 못 채우면 quality 높은 순으로 추가
    if len(selected) < min_biweeks:
        remaining = avg_quality[~avg_quality['biweek'].isin(selected)]
        remaining = remaining.nlargest(min_biweeks - len(selected), 'quality')
        selected.extend(remaining['biweek'].tolist())
        print(f"[Step 3] 최소 개수 채우기 후: {len(selected)}개")
    
    # 4. 최대 개수 초과하면 quality 낮은 것 제거
    if len(selected) > max_biweeks:
        selected_df = avg_quality[avg_quality['biweek'].isin(selected)]
        selected_df = selected_df.nlargest(max_biweeks, 'quality')
        selected = selected_df['biweek'].tolist()
        print(f"[Step 4] 최대 개수로 제한: {len(selected)}개")
    
    # 정렬
    selected = sorted(set(selected))
    
    # ========================================
    # 결과 출력
    # ========================================
    print("\n" + "="*70)
    print(f"FINAL SELECTED BIWEEKS: {len(selected)}개")
    print("="*70)
    print(f"Biweek indices: {selected}")
    print("\nDetails:")
    print(f"{'Biweek':<15} {'Stage':<15} {'S1 %':>8} {'S2 %':>8} {'Quality':>10}")
    print("-"*60)
    
    total_s1 = 0
    total_s2 = 0
    total_quality = 0
    
    for biweek in selected:
        row = avg_quality[avg_quality['biweek'] == biweek].iloc[0]
        print(f"{row['biweek_label']:<15} {row['stage']:<15} {row['s1_pct']:>8.1f} {row['s2_pct']:>8.1f} {row['quality']:>10.1f}")
        total_s1 += row['s1_pct']
        total_s2 += row['s2_pct']
        total_quality += row['quality']
    
    print("-"*60)
    n = len(selected)
    print(f"{'AVERAGE':<15} {'':<15} {total_s1/n:>8.1f} {total_s2/n:>8.1f} {total_quality/n:>10.1f}")
    
    # ========================================
    # 년도별 검증: 선택된 biweeks의 quality 확인
    # ========================================
    print("\n" + "="*70)
    print("YEAR-BY-YEAR QUALITY FOR SELECTED BIWEEKS")
    print("="*70)
    
    for year in sorted(years):
        df_year = df_quality[df_quality['year'] == year]
        selected_quality = df_year[df_year['biweek'].isin(selected)]['quality']
        avg_q = selected_quality.mean()
        min_q = selected_quality.min()
        max_q = selected_quality.max()
        print(f"Year {year}: Avg={avg_q:.1f}, Min={min_q:.1f}, Max={max_q:.1f}")
    
    # ========================================
    # 제외된 biweeks 정보
    # ========================================
    excluded = sorted(set(range(12)) - set(selected))
    if excluded:
        print("\n" + "="*70)
        print(f"EXCLUDED BIWEEKS: {len(excluded)}개")
        print("="*70)
        print(f"Biweek indices: {excluded}")
        print("\nDetails:")
        for biweek in excluded:
            row = avg_quality[avg_quality['biweek'] == biweek].iloc[0]
            print(f"  {row['biweek_label']:<15} {row['stage']:<15} Quality={row['quality']:.1f}")
    
    # ========================================
    # 결과 저장
    # ========================================
    if output_path:
        result_df = avg_quality.copy()
        result_df['selected'] = result_df['biweek'].isin(selected)
        result_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return selected, avg_quality


def compare_selection_strategies(csv_path):
    """
    여러 선택 전략 비교
    """
    print("\n" + "="*70)
    print("COMPARISON OF SELECTION STRATEGIES")
    print("="*70)
    
    strategies = [
        ("All 12 biweeks", 12, 12),
        ("Best 10 biweeks", 10, 10),
        ("Best 8 biweeks", 8, 8),
        ("Best 6 biweeks (Jun-Sep focus)", 6, 6),
    ]
    
    results = []
    
    for name, min_b, max_b in strategies:
        print(f"\n--- {name} ---")
        selected, quality_df = select_best_biweeks_across_years(
            csv_path, 
            min_biweeks=min_b, 
            max_biweeks=max_b
        )
        
        # 선택된 biweeks의 평균 quality
        selected_quality = quality_df[quality_df['biweek'].isin(selected)]
        avg_q = selected_quality['quality'].mean()
        avg_s1 = selected_quality['s1_pct'].mean()
        avg_s2 = selected_quality['s2_pct'].mean()
        
        results.append({
            'strategy': name,
            'n_biweeks': len(selected),
            'selected': selected,
            'avg_quality': avg_q,
            'avg_s1': avg_s1,
            'avg_s2': avg_s2
        })
    
    print("\n" + "="*70)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Strategy':<30} {'N':>5} {'Avg S1':>10} {'Avg S2':>10} {'Avg Quality':>12}")
    print("-"*70)
    
    for r in results:
        print(f"{r['strategy']:<30} {r['n_biweeks']:>5} {r['avg_s1']:>10.1f} {r['avg_s2']:>10.1f} {r['avg_quality']:>12.1f}")
    
    return results


def print_biweek_labels(selected_biweeks):
    """선택된 biweeks의 label 출력"""
    biweek_labels = [
        'Apr 1-15', 'Apr 16-30',
        'May 1-15', 'May 16-31',
        'Jun 1-15', 'Jun 16-30',
        'Jul 1-15', 'Jul 16-31',
        'Aug 1-15', 'Aug 16-31',
        'Sep 1-15', 'Sep 16-30'
    ]
    
    print("\nSelected biweek labels:")
    for b in selected_biweeks:
        print(f"  Biweek {b}: {biweek_labels[b]}")


if __name__ == "__main__":
    csv_path = 'biweek_availability_stats_all_years.csv'
    
    # 기본 선택 (8-10개 권장)
    selected_biweeks, quality_df = select_best_biweeks_across_years(
        csv_path,
        min_biweeks=8,
        max_biweeks=8,
        output_path='selected_biweeks_result.csv'
    )
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Selected biweeks: {selected_biweeks}")
    print(f"Total: {len(selected_biweeks)} biweeks")
    print_biweek_labels(selected_biweeks)
    
    # 선택 안 된 biweeks도 출력
    all_biweeks = set(range(12))
    excluded = sorted(all_biweeks - set(selected_biweeks))
    print(f"\nExcluded biweeks: {excluded}")
    if excluded:
        print_biweek_labels(excluded)
    
    # 여러 전략 비교 (optional)
    print("\n\n")
    compare_selection_strategies(csv_path)