#!/usr/bin/env python3
"""
FINAL TRUTH ANALYSIS
====================
This script will tell you EXACTLY what happened with your data.
No placeholders, no guessing - just facts from your actual files.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import os

print("="*80)
print("FINAL TRUTH ANALYSIS - LET'S FIGURE THIS OUT ONCE AND FOR ALL")
print("="*80)

# 1. COUNT EVERYTHING
print("\n1. COUNTING ALL FILES:")
print("-" * 40)

# Count raw charts
raw_charts_path = Path("data/raw_charts")
raw_chart_files = list(raw_charts_path.glob("*.png"))
print(f"Raw chart files found: {len(raw_chart_files)}")

# Count perturbations
pert_path = Path("data/perturbations")
pert_files = list(pert_path.glob("*.png"))
print(f"Perturbation files found: {len(pert_files)}")

# Load extraction results
extraction_path = "E:/langchain/Dissertation/data/analysis_cache/complete_extraction_results.json"
with open(extraction_path, 'r') as f:
    extractions = json.load(f)
print(f"Extraction results found: {len(extractions)}")

# Load robustness analysis
robustness_path = "E:/langchain/Dissertation/data/analysis_cache/robustness_analysis_corrected.csv"
robustness_df = pd.read_csv(robustness_path)
print(f"Robustness evaluations: {len(robustness_df)}")

# 2. ANALYZE ORIGINAL CHARTS
print("\n2. ANALYZING ORIGINAL CHARTS:")
print("-" * 40)

# Get all chart IDs from raw charts
chart_ids_from_files = set()
for f in raw_chart_files:
    parts = f.stem.split('_')
    if len(parts) >= 2:
        chart_id = f"{parts[0]}_{parts[1]}"
        chart_ids_from_files.add(chart_id)

print(f"Unique chart IDs from files: {len(chart_ids_from_files)}")

# Expected chart IDs (0-199 or 1-200?)
expected_charts = [f"chart_{i:03d}" for i in range(200)]
print(f"Expected chart IDs: {len(expected_charts)}")

# Find missing chart files
missing_chart_files = [c for c in expected_charts if c not in chart_ids_from_files]
print(f"Missing chart files: {len(missing_chart_files)}")
if missing_chart_files:
    print(f"  Missing: {missing_chart_files[:5]}...")

# 3. ANALYZE PERTURBATIONS PER CHART
print("\n3. PERTURBATIONS PER CHART:")
print("-" * 40)

# Count perturbations per chart
pert_per_chart = defaultdict(int)
pert_types_per_chart = defaultdict(set)

for pert_file in pert_files:
    parts = pert_file.stem.split('_')
    if len(parts) >= 2:
        chart_id = f"{parts[0]}_{parts[1]}"
        pert_per_chart[chart_id] += 1
        
        # Get perturbation type
        if len(parts) >= 5:
            pert_type = '_'.join(parts[4:-1])
            pert_types_per_chart[chart_id].add(pert_type)

# Find charts with no perturbations
charts_with_no_pert = [c for c in expected_charts if c not in pert_per_chart]
print(f"Charts with NO perturbations: {len(charts_with_no_pert)}")

# Show perturbation distribution
pert_counts = list(pert_per_chart.values())
if pert_counts:
    print(f"Average perturbations per chart: {sum(pert_counts)/len(pert_counts):.1f}")
    print(f"Max perturbations on a chart: {max(pert_counts)}")
    print(f"Min perturbations on a chart: {min(pert_counts)}")

# 4. ANALYZE EXTRACTIONS
print("\n4. EXTRACTION ANALYSIS:")
print("-" * 40)

# Categorize extraction keys
original_extractions = []
perturbation_extractions = []

for key in extractions.keys():
    if key.endswith('_original'):
        original_extractions.append(key)
    else:
        perturbation_extractions.append(key)

print(f"Original extractions: {len(original_extractions)}")
print(f"Perturbation extractions: {len(perturbation_extractions)}")

# 5. FIND THE 52 CHARTS
print("\n5. THE 52 CHARTS WITHOUT PERTURBATIONS:")
print("-" * 40)

# These are charts that exist but have no perturbations
charts_no_pert_reasons = []

for chart_id in charts_with_no_pert:
    reason = "Unknown"
    
    # Check if chart file exists
    chart_exists = any(f.stem.startswith(chart_id) for f in raw_chart_files)
    
    if not chart_exists:
        reason = "Chart file doesn't exist"
    else:
        # Check if original extraction exists
        orig_key = None
        for key in original_extractions:
            if key.startswith(chart_id):
                orig_key = key
                break
        
        if not orig_key:
            reason = "No original extraction attempted"
        else:
            # Check extraction quality
            extraction = extractions.get(orig_key)
            if extraction is None:
                reason = "Original extraction returned null"
            elif 'error' in extraction:
                reason = f"Extraction error: {extraction.get('error', '')[:50]}"
            elif 'data' not in extraction:
                reason = "Extraction missing data field"
            elif not extraction.get('data'):
                reason = "Extraction has empty data"
            else:
                reason = "Unknown - extraction seems valid but no perturbations generated"
    
    charts_no_pert_reasons.append({
        'Chart ID': chart_id,
        'Reason': reason
    })

df_52 = pd.DataFrame(charts_no_pert_reasons)
print(f"Found {len(df_52)} charts without perturbations")
print("\nReasons breakdown:")
print(df_52['Reason'].value_counts())

# 6. FIND THE 952 EXCLUDED PERTURBATIONS
print("\n6. THE 952 EXCLUDED PERTURBATIONS:")
print("-" * 40)

# Get all perturbation keys that were evaluated
evaluated_keys = set(robustness_df['extraction_key'].unique())
evaluated_pert_keys = [k for k in evaluated_keys if not k.endswith('_original')]

print(f"Perturbations evaluated: {len(evaluated_pert_keys)}")

# Find which perturbation files were NOT evaluated
excluded_perturbations = []

for pert_file in pert_files:
    filename = pert_file.stem
    
    if filename not in evaluated_pert_keys:
        # This perturbation was not evaluated
        parts = filename.split('_')
        
        # Parse the filename
        if len(parts) >= 6:
            chart_id = f"{parts[0]}_{parts[1]}"
            complexity = parts[2]
            chart_type = parts[3]
            pert_type = '_'.join(parts[4:-1])
            intensity = parts[-1]
            
            # Check if extraction exists
            reason = "Unknown"
            if filename in extractions:
                extraction = extractions[filename]
                if extraction is None:
                    reason = "Extraction is null"
                elif 'error' in extraction:
                    reason = f"Error: {extraction.get('error', '')[:50]}"
                elif 'data' not in extraction:
                    reason = "Missing data field"
                elif not extraction.get('data'):
                    reason = "Empty data extracted"
                else:
                    reason = "Valid extraction but excluded from evaluation"
            else:
                reason = "No extraction found for this perturbation"
            
            excluded_perturbations.append({
                'Filename': filename,
                'Chart ID': chart_id,
                'Chart Type': chart_type,
                'Complexity': complexity,
                'Perturbation Type': pert_type,
                'Intensity': intensity,
                'Exclusion Reason': reason
            })

df_952 = pd.DataFrame(excluded_perturbations)
print(f"Found {len(df_952)} excluded perturbations")

print("\nExclusion reasons breakdown:")
print(df_952['Exclusion Reason'].value_counts())

print("\nPerturbation types excluded most:")
print(df_952['Perturbation Type'].value_counts().head(10))

# 7. FINAL SUMMARY
print("\n" + "="*80)
print("FINAL SUMMARY - THE TRUTH:")
print("="*80)

print(f"""
1. You have {len(raw_chart_files)} chart files (expected 200)
2. You have {len(pert_files)} perturbation files (expected ~1,650)
3. {len(charts_with_no_pert)} charts have NO perturbations (expected 52)
4. {len(df_952)} perturbations were not evaluated (expected 952)

WHAT HAPPENED:
- Some charts failed original extraction → no perturbations could be generated
- Some perturbations were generated but extraction failed → excluded from evaluation
- Some had valid extractions but were excluded (possibly quality issues)
""")

# Save the REAL data
df_52.to_csv("FINAL_52_charts_no_perturbations.csv", index=False)
df_952.to_csv("FINAL_952_excluded_perturbations.csv", index=False)

print("\nSAVED FINAL RESULTS:")
print("- FINAL_52_charts_no_perturbations.csv")
print("- FINAL_952_excluded_perturbations.csv")
print("\nTHESE ARE YOUR REAL NUMBERS WITH REAL REASONS!")

# Extra validation
print("\n8. VALIDATION:")
print("-" * 40)
print(f"Do the numbers add up?")
print(f"- Charts with perturbations: {len(pert_per_chart)}")
print(f"- Charts without perturbations: {len(charts_with_no_pert)}")
print(f"- Total: {len(pert_per_chart) + len(charts_with_no_pert)} (should be ~200)")
print(f"\n- Perturbations evaluated: {len(evaluated_pert_keys)}")
print(f"- Perturbations excluded: {len(df_952)}")
print(f"- Total: {len(evaluated_pert_keys) + len(df_952)} (should be ~1,650)")

print("\n ANALYSIS COMPLETE!")