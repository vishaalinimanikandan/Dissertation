#!/usr/bin/env python3
"""
Investigate why we have 203 charts instead of 200
"""

import os
from pathlib import Path
import re

def investigate_charts():
    """Find out what the 3 extra charts are"""
    print("=== INVESTIGATING 203 CHARTS ===\n")
    
    raw_charts_dir = Path("data/raw_charts")
    chart_files = sorted(list(raw_charts_dir.glob("*.png")))
    
    print(f"Total charts found: {len(chart_files)}\n")
    
    # Extract chart IDs
    chart_ids = []
    chart_info = []
    
    for chart_file in chart_files:
        filename = chart_file.stem
        parts = filename.split('_')
        
        if len(parts) >= 2:
            chart_num = parts[1]  # The number part
            try:
                num = int(chart_num)
                chart_ids.append(num)
                chart_info.append({
                    'file': chart_file.name,
                    'number': num,
                    'full_id': f"{parts[0]}_{parts[1]}"
                })
            except ValueError:
                print(f"  Unusual chart ID: {filename}")
    
    # Check for missing numbers in sequence
    chart_ids_sorted = sorted(chart_ids)
    expected_ids = list(range(0, 200))  # 0-199 or 1-200?
    
    print("CHART ID ANALYSIS:")
    print(f"Lowest ID: {min(chart_ids_sorted)}")
    print(f"Highest ID: {max(chart_ids_sorted)}")
    print(f"Total unique IDs: {len(set(chart_ids))}")
    
    # Find duplicates
    duplicates = []
    seen = set()
    for num in chart_ids:
        if num in seen:
            duplicates.append(num)
        seen.add(num)
    
    if duplicates:
        print(f"\n  DUPLICATES FOUND: {duplicates}")
        for dup in duplicates:
            print(f"\nDuplicate chart_{dup:03d}:")
            for info in chart_info:
                if info['number'] == dup:
                    print(f"  - {info['file']}")
    
    # Find gaps in sequence
    missing_ids = []
    for i in range(min(chart_ids_sorted), max(chart_ids_sorted) + 1):
        if i not in chart_ids_sorted:
            missing_ids.append(i)
    
    if missing_ids:
        print(f"\n  MISSING IDs: {missing_ids}")
    
    # Find IDs outside expected range
    if min(chart_ids_sorted) < 0:
        print(f"\n IDs below 0: {[id for id in chart_ids_sorted if id < 0]}")
    
    extra_ids = [id for id in chart_ids_sorted if id >= 200]
    if extra_ids:
        print(f"\n  IDs 200 or above: {extra_ids}")
        for extra_id in extra_ids:
            for info in chart_info:
                if info['number'] == extra_id:
                    print(f"  - {info['file']}")
    
    # Show first and last few files
    print("\nFIRST 5 CHARTS:")
    for info in chart_info[:5]:
        print(f"  {info['file']}")
    
    print("\nLAST 5 CHARTS:")
    for info in chart_info[-5:]:
        print(f"  {info['file']}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    if len(set(chart_ids)) < len(chart_ids):
        print("✓ Found duplicate chart IDs")
    if extra_ids:
        print(f"✓ Found {len(extra_ids)} charts with ID >= 200")
    if missing_ids:
        print(f"✓ Found {len(missing_ids)} missing IDs in sequence")
    
    return chart_info

if __name__ == "__main__":
    investigate_charts()