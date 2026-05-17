"""
Threshold Sweep 결과 분석 스크립트
"""

import pandas as pd
from pathlib import Path
import sys

def analyze_threshold_sweep(base_dir):
    """Threshold sweep 결과 수집 및 분석"""
    
    base = Path(base_dir)
    results = []
    
    # 각 threshold 결과 수집
    for thresh_dir in sorted(base.glob('thresh_*')):
        thresh_val = float(thresh_dir.name.replace('thresh_', ''))
        
        # Overall 결과
        overall_file = thresh_dir / 'RF-DETR-Medium' / 'detection_overall.csv'
        if overall_file.exists():
            df_overall = pd.read_csv(overall_file)
            
            # Per-class 결과
            perclass_file = thresh_dir / 'RF-DETR-Medium' / 'detection_per_class.csv'
            df_perclass = pd.read_csv(perclass_file)
            
            # Caries 결과 추출
            caries_row = df_perclass[df_perclass['category_name'] == 'Caries'].iloc[0]
            abrasion_row = df_perclass[df_perclass['category_name'] == 'Abrasion'].iloc[0]
            
            results.append({
                'threshold': thresh_val,
                'overall_precision': df_overall['precision'].iloc[0],
                'overall_recall': df_overall['recall'].iloc[0],
                'overall_f1': df_overall['f1'].iloc[0],
                'overall_map50': df_overall['map_50'].iloc[0],
                'overall_map5095': df_overall['map_50_95'].iloc[0],
                'caries_precision': caries_row['precision'],
                'caries_recall': caries_row['recall'],
                'caries_f1': caries_row['f1'],
                'caries_map50': caries_row['map_50'],
                'abrasion_precision': abrasion_row['precision'],
                'abrasion_recall': abrasion_row['recall'],
                'abrasion_f1': abrasion_row['f1'],
            })
    
    df = pd.DataFrame(results)
    
    # 결과 출력
    print("\n" + "="*80)
    print("THRESHOLD SWEEP RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    # 최적 threshold 찾기
    print("\n" + "="*80)
    print("BEST THRESHOLDS")
    print("="*80)
    
    best_f1_idx = df['overall_f1'].idxmax()
    best_caries_recall_idx = df['caries_recall'].idxmax()
    best_caries_f1_idx = df['caries_f1'].idxmax()
    
    print(f"\nBest Overall F1-Score:")
    print(f"  Threshold: {df.loc[best_f1_idx, 'threshold']}")
    print(f"  F1: {df.loc[best_f1_idx, 'overall_f1']:.4f}")
    print(f"  Precision: {df.loc[best_f1_idx, 'overall_precision']:.4f}")
    print(f"  Recall: {df.loc[best_f1_idx, 'overall_recall']:.4f}")
    
    print(f"\nBest Caries Recall:")
    print(f"  Threshold: {df.loc[best_caries_recall_idx, 'threshold']}")
    print(f"  Caries Recall: {df.loc[best_caries_recall_idx, 'caries_recall']:.4f}")
    print(f"  Caries F1: {df.loc[best_caries_recall_idx, 'caries_f1']:.4f}")
    
    print(f"\nBest Caries F1-Score:")
    print(f"  Threshold: {df.loc[best_caries_f1_idx, 'threshold']}")
    print(f"  Caries F1: {df.loc[best_caries_f1_idx, 'caries_f1']:.4f}")
    print(f"  Caries Recall: {df.loc[best_caries_f1_idx, 'caries_recall']:.4f}")
    print(f"  Caries Precision: {df.loc[best_caries_f1_idx, 'caries_precision']:.4f}")
    
    # CSV 저장
    output_file = base / 'threshold_sweep_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = '/data/thdud23/outputs/thresh_sweep'
    
    df = analyze_threshold_sweep(base_dir)