#!/usr/bin/env python3
"""
Fix Duplicate Charts Script
===========================
1. Investigates which version of duplicates was used in analysis
2. Fixes duplicates by renumbering them
3. Updates all related perturbations
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

def investigate_duplicate_usage():
    """Check which version of duplicates was actually used"""
    print("=== INVESTIGATING DUPLICATE USAGE ===\n")
    
    # Load extraction results to see which versions were extracted
    extraction_path = "E:/langchain/Dissertation/data/analysis_cache/complete_extraction_results.json"
    
    duplicates = {
        'chart_001': ['chart_001_complex_bar', 'chart_001_medium_bar'],
        'chart_002': ['chart_002_medium_bar', 'chart_002_medium_line'],
        'chart_003': ['chart_003_advanced_pie', 'chart_003_medium_bar']
    }
    
    if os.path.exists(extraction_path):
        with open(extraction_path, 'r') as f:
            extractions = json.load(f)
        
        print("Checking extraction results for duplicates:\n")
        
        for chart_id, versions in duplicates.items():
            print(f"{chart_id}:")
            for version in versions:
                # Check original extraction
                original_key = f"{version}_original"
                if original_key in extractions:
                    print(f"  ✓ {version} - FOUND in extractions")
                else:
                    print(f"  ✗ {version} - NOT in extractions")
                
                # Check perturbations
                pert_count = sum(1 for key in extractions.keys() if key.startswith(f"{version}_") and not key.endswith('_original'))
                if pert_count > 0:
                    print(f"    → Has {pert_count} perturbations")
            print()
    
    # Check perturbation files
    print("\nChecking perturbation files:\n")
    pert_dir = Path("data/perturbations")
    
    for chart_id, versions in duplicates.items():
        print(f"{chart_id}:")
        for version in versions:
            pert_files = list(pert_dir.glob(f"{version}_*.png"))
            print(f"  {version}: {len(pert_files)} perturbation files")
        print()
    
    return duplicates

def fix_duplicates():
    """Fix duplicates by renumbering"""
    print("\n=== FIXING DUPLICATES ===\n")
    
    # Strategy: Keep the first version, renumber the second version to 201, 202, 203
    renumber_map = {
        'chart_001_medium_bar': 'chart_201_medium_bar',
        'chart_002_medium_line': 'chart_202_medium_line', 
        'chart_003_medium_bar': 'chart_203_medium_bar'
    }
    
    # Also handle chart_200 → chart_204 to keep sequence clean
    renumber_map['chart_200_complex_pie'] = 'chart_204_complex_pie'
    
    print("Renumbering plan:")
    for old, new in renumber_map.items():
        print(f"  {old} → {new}")
    
    # Create backup directory
    backup_dir = Path("data/backup_before_fix")
    backup_dir.mkdir(exist_ok=True)
    
    # Process raw charts
    print("\n1. Fixing raw charts...")
    raw_dir = Path("data/raw_charts")
    fixed_count = 0
    
    for old_name, new_name in renumber_map.items():
        old_file = raw_dir / f"{old_name}.png"
        new_file = raw_dir / f"{new_name}.png"
        
        if old_file.exists():
            # Backup
            shutil.copy2(old_file, backup_dir / f"{old_name}.png")
            # Rename
            shutil.move(str(old_file), str(new_file))
            print(f"  ✓ Renamed {old_name} → {new_name}")
            fixed_count += 1
    
    # Process perturbations
    print("\n2. Fixing perturbations...")
    pert_dir = Path("data/perturbations")
    pert_fixed = 0
    
    for old_base, new_base in renumber_map.items():
        # Find all perturbations for this chart
        pert_files = list(pert_dir.glob(f"{old_base}_*.png"))
        
        for pert_file in pert_files:
            # Get the perturbation suffix
            old_name = pert_file.stem
            suffix = old_name.replace(old_base, '')
            new_name = f"{new_base}{suffix}"
            
            old_path = pert_file
            new_path = pert_dir / f"{new_name}.png"
            
            # Backup and rename
            shutil.copy2(old_path, backup_dir / pert_file.name)
            shutil.move(str(old_path), str(new_path))
            pert_fixed += 1
        
        if pert_files:
            print(f"  ✓ Fixed {len(pert_files)} perturbations for {old_base}")
    
    print(f"\n✓ Fixed {fixed_count} raw charts")
    print(f"✓ Fixed {pert_fixed} perturbations")
    print(f"✓ Backups saved to: {backup_dir}")
    
    return renumber_map

def update_organized_folders():
    """Update the organized folders with fixed names"""
    print("\n3. Updating organized folders...")
    
    # Re-run organization with fixed files
    os.system("python reorganize_charts_script.py")

def verify_fix():
    """Verify no more duplicates exist"""
    print("\n=== VERIFYING FIX ===\n")
    
    raw_dir = Path("data/raw_charts")
    chart_files = list(raw_dir.glob("chart_*.png"))
    
    # Extract all chart IDs
    chart_ids = []
    for f in chart_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            try:
                chart_ids.append(int(parts[1]))
            except ValueError:
                pass
    
    # Check for duplicates
    seen = set()
    duplicates = []
    for id in chart_ids:
        if id in seen:
            duplicates.append(id)
        seen.add(id)
    
    print(f"Total charts: {len(chart_files)}")
    print(f"Unique IDs: {len(set(chart_ids))}")
    print(f"ID range: {min(chart_ids)} - {max(chart_ids)}")
    
    if duplicates:
        print(f"  Still have duplicates: {duplicates}")
    else:
        print(" No duplicates found!")
    
    print(f"\nExpected: 204 charts (200 original + 4 renumbered)")
    print(f"Actual: {len(chart_files)} charts")

def create_mapping_file(renumber_map):
    """Create a mapping file for future reference"""
    mapping_data = {
        'date': str(Path.ctime(Path.cwd())),
        'duplicates_fixed': renumber_map,
        'explanation': {
            'chart_001': 'Kept complex_bar, renamed medium_bar to 201',
            'chart_002': 'Kept medium_bar, renamed medium_line to 202',
            'chart_003': 'Kept advanced_pie, renamed medium_bar to 203',
            'chart_200': 'Renamed to 204 to avoid confusion'
        }
    }
    
    with open('data/duplicate_fix_mapping.json', 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print("\n✓ Mapping saved to: data/duplicate_fix_mapping.json")

def main():
    """Main execution"""
    print("="*60)
    print("DUPLICATE CHARTS FIX")
    print("="*60)
    
    # First investigate which versions were used
    investigate_duplicate_usage()
    
    # Ask for confirmation
    response = input("\nProceed with fixing duplicates? (y/n): ")
    
    if response.lower() == 'y':
        # Fix duplicates
        renumber_map = fix_duplicates()
        
        # Create mapping file
        create_mapping_file(renumber_map)
        
        # Verify fix
        verify_fix()
        
        print("\n FIX COMPLETE!")
        print("\nNext steps:")
        print("1. Re-run organization script to update organized folders")
        print("2. Update any analysis that referenced the old chart IDs")
        print("3. Document this fix in your methodology")
    else:
        print("\nFix cancelled.")

if __name__ == "__main__":
    main()