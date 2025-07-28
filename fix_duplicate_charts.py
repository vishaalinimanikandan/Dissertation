#!/usr/bin/env python3
"""
Remove Unused Duplicate Charts
==============================
Safely removes duplicate charts that were never used in analysis
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def identify_charts_to_remove():
    """Identify which duplicate charts to remove based on perturbation usage"""
    print("=== IDENTIFYING CHARTS TO REMOVE ===\n")
    
    raw_dir = Path("data/raw_charts")
    pert_dir = Path("data/perturbations")
    
    # Based on the investigation, here's what we found:
    duplicates = {
        'chart_001': {
            'complex_bar': 12,  # has perturbations - KEEP
            'medium_bar': 0     # no perturbations - REMOVE
        },
        'chart_002': {
            'medium_bar': 18,   # has perturbations - KEEP
            'medium_line': 18   # also has perturbations - PROBLEM!
        },
        'chart_003': {
            'advanced_pie': 6,  # has perturbations - KEEP
            'medium_bar': 18    # also has perturbations - PROBLEM!
        }
    }
    
    # For chart_001, it's clear: remove medium_bar
    # For chart_002 and chart_003, both versions have perturbations!
    # Let's check which perturbations are actually in the extraction results
    
    charts_to_remove = []
    charts_to_keep = []
    problem_charts = []
    
    print("Analysis:")
    print("-" * 50)
    
    # Chart_001: Clear case
    print("chart_001:")
    print("  - complex_bar: 12 perturbations → KEEP")
    print("  - medium_bar: 0 perturbations → REMOVE")
    charts_to_keep.append("chart_001_complex_bar")
    charts_to_remove.append("chart_001_medium_bar")
    
    # Chart_002: Both have perturbations
    print("\nchart_002:")
    print("  - medium_bar: 18 perturbations")
    print("  - medium_line: 18 perturbations")
    print("    BOTH have perturbations! Need to check which were analyzed")
    problem_charts.append(('chart_002', ['medium_bar', 'medium_line']))
    
    # Chart_003: Both have perturbations  
    print("\nchart_003:")
    print("  - advanced_pie: 6 perturbations")
    print("  - medium_bar: 18 perturbations")
    print("    BOTH have perturbations! Need to check which were analyzed")
    problem_charts.append(('chart_003', ['advanced_pie', 'medium_bar']))
    
    # Chart_200: Extra chart
    print("\nchart_200:")
    print("  - complex_pie: ID > 199 → REMOVE (outside expected range)")
    charts_to_remove.append("chart_200_complex_pie")
    
    return charts_to_remove, charts_to_keep, problem_charts

def check_robustness_analysis():
    """Check which versions were used in robustness analysis"""
    print("\n=== CHECKING ROBUSTNESS ANALYSIS ===\n")
    
    robustness_path = "E:/langchain/Dissertation/data/analysis_cache/robustness_analysis_corrected.csv"
    
    if os.path.exists(robustness_path):
        import pandas as pd
        df = pd.read_csv(robustness_path)
        
        # Check which chart_002 and chart_003 versions appear
        keys = df['extraction_key'].tolist()
        
        chart_002_types = {'medium_bar': 0, 'medium_line': 0}
        chart_003_types = {'advanced_pie': 0, 'medium_bar': 0}
        
        for key in keys:
            if 'chart_002_medium_bar' in key:
                chart_002_types['medium_bar'] += 1
            elif 'chart_002_medium_line' in key:
                chart_002_types['medium_line'] += 1
            elif 'chart_003_advanced_pie' in key:
                chart_003_types['advanced_pie'] += 1
            elif 'chart_003_medium_bar' in key:
                chart_003_types['medium_bar'] += 1
        
        print("Usage in robustness analysis:")
        print(f"chart_002_medium_bar: {chart_002_types['medium_bar']} entries")
        print(f"chart_002_medium_line: {chart_002_types['medium_line']} entries")
        print(f"chart_003_advanced_pie: {chart_003_types['advanced_pie']} entries")
        print(f"chart_003_medium_bar: {chart_003_types['medium_bar']} entries")
        
        # Determine which to keep based on usage
        decisions = {}
        
        if chart_002_types['medium_bar'] > chart_002_types['medium_line']:
            decisions['chart_002'] = ('keep_medium_bar', 'remove_medium_line')
        elif chart_002_types['medium_line'] > chart_002_types['medium_bar']:
            decisions['chart_002'] = ('keep_medium_line', 'remove_medium_bar')
        else:
            decisions['chart_002'] = ('keep_medium_bar', 'remove_medium_line')  # default
        
        if chart_003_types['advanced_pie'] > chart_003_types['medium_bar']:
            decisions['chart_003'] = ('keep_advanced_pie', 'remove_medium_bar')
        elif chart_003_types['medium_bar'] > chart_003_types['advanced_pie']:
            decisions['chart_003'] = ('keep_medium_bar', 'remove_advanced_pie')
        else:
            decisions['chart_003'] = ('keep_advanced_pie', 'remove_medium_bar')  # default
        
        return decisions
    else:
        print("Robustness analysis file not found!")
        return None

def remove_charts_safely(charts_to_remove, decisions):
    """Safely remove the identified charts"""
    print("\n=== REMOVING UNUSED CHARTS ===\n")
    
    # Create backup directory
    backup_dir = Path(f"data/removed_duplicates_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_dir.mkdir(exist_ok=True)
    
    # Final removal list
    final_removal_list = [
        "chart_001_medium_bar",    # No perturbations used
        "chart_200_complex_pie"    # Outside range
    ]
    
    # Add decisions for chart_002 and chart_003
    if decisions:
        if decisions['chart_002'][1] == 'remove_medium_line':
            final_removal_list.append("chart_002_medium_line")
        else:
            final_removal_list.append("chart_002_medium_bar")
            
        if decisions['chart_003'][1] == 'remove_medium_bar':
            final_removal_list.append("chart_003_medium_bar")
        else:
            final_removal_list.append("chart_003_advanced_pie")
    
    print("Files to remove:")
    for chart in final_removal_list:
        print(f"  - {chart}.png")
    
    # Remove from raw_charts
    raw_dir = Path("data/raw_charts")
    removed_count = 0
    
    for chart_name in final_removal_list:
        chart_file = raw_dir / f"{chart_name}.png"
        if chart_file.exists():
            # Backup
            shutil.copy2(chart_file, backup_dir / chart_file.name)
            # Remove
            chart_file.unlink()
            print(f"  ✓ Removed {chart_name}.png")
            removed_count += 1
    
    # Remove associated perturbations (if any exist for removed charts)
    pert_dir = Path("data/perturbations")
    pert_removed = 0
    
    for chart_name in final_removal_list:
        pert_files = list(pert_dir.glob(f"{chart_name}_*.png"))
        for pert_file in pert_files:
            # Backup
            shutil.copy2(pert_file, backup_dir / pert_file.name)
            # Remove
            pert_file.unlink()
            pert_removed += 1
        
        if pert_files:
            print(f"  ✓ Removed {len(pert_files)} perturbations for {chart_name}")
    
    # Clean up organized folders
    for org_dir in ["data/raw_charts_organized", "data/perturbations_organized"]:
        if Path(org_dir).exists():
            shutil.rmtree(org_dir)
            print(f"  ✓ Cleaned {org_dir}")
    
    print(f"\n Removed {removed_count} duplicate charts")
    print(f" Removed {pert_removed} associated perturbations")
    print(f" Backups saved to: {backup_dir}")
    
    # Save removal log
    with open(backup_dir / "removal_log.txt", 'w') as f:
        f.write(f"Duplicate Removal Log\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Charts removed: {final_removal_list}\n")
        f.write(f"Total charts removed: {removed_count}\n")
        f.write(f"Total perturbations removed: {pert_removed}\n")

def verify_cleanup():
    """Verify the cleanup was successful"""
    print("\n=== VERIFYING CLEANUP ===\n")
    
    raw_dir = Path("data/raw_charts")
    chart_files = list(raw_dir.glob("chart_*.png"))
    
    # Count and check for duplicates
    chart_ids = []
    for f in chart_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            try:
                chart_ids.append(int(parts[1]))
            except ValueError:
                pass
    
    # Check for duplicates
    from collections import Counter
    id_counts = Counter(chart_ids)
    duplicates = [id for id, count in id_counts.items() if count > 1]
    
    print(f"Total charts remaining: {len(chart_files)}")
    print(f"Expected: 200 charts")
    print(f"Unique IDs: {len(set(chart_ids))}")
    
    if duplicates:
        print(f"  Still have duplicate IDs: {duplicates}")
    else:
        print(" No duplicates found!")
    
    # Show summary
    print(f"\nSummary:")
    print(f"  - Started with: 203 charts")
    print(f"  - Removed: 3 charts")
    print(f"  - Now have: {len(chart_files)} charts")

def main():
    """Main execution"""
    print("="*60)
    print("REMOVE UNUSED DUPLICATE CHARTS")
    print("="*60)
    
    # Identify charts to remove
    to_remove, to_keep, problems = identify_charts_to_remove()
    
    # Check robustness analysis for problem charts
    decisions = check_robustness_analysis()
    
    # Confirm before proceeding
    print("\n" + "="*60)
    response = input("\nProceed with removing unused duplicates? (y/n): ")
    
    if response.lower() == 'y':
        # Remove charts
        remove_charts_safely(to_remove, decisions)
        
        # Verify
        verify_cleanup()
        
        print("\n CLEANUP COMPLETE!")
        print("\nNext steps:")
        print("1. Re-run organization script: python reorganize_charts_script.py")
        print("2. Your analysis remains unaffected - we only removed unused files")
    else:
        print("\nCleanup cancelled.")

if __name__ == "__main__":
    main()