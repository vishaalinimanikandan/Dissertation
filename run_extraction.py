# run_full_extraction.py - MAXIMIZE $5 BUDGET

import os
import json
import base64
import pandas as pd
import numpy as np
from PIL import Image
import openai
from dotenv import load_dotenv
from pathlib import Path
import time
import random
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class GPT4VisionExtractor:
    """Professional GPT-4V extraction system"""
    
    def __init__(self, client):
        self.client = client
        self.extraction_prompt = """
You are a professional data analyst. Extract numerical values and associated labels from this chart image with maximum precision.

Return ONLY a valid JSON object in this exact format:
{
  "chart_title": "Exact title from the chart",
  "chart_type": "bar/pie/line/scatter/area/stacked_bar/grouped_bar",
  "data": [
    {"category": "Category_1", "value": numeric_value},
    {"category": "Category_2", "value": numeric_value}
  ],
  "extraction_confidence": "high/medium/low",
  "notes": "Any issues or observations"
}

Requirements:
- Extract ALL visible data points
- Use exact category names from the chart
- Preserve numerical precision
- For pie charts, ensure percentages sum to ~100%
- Report confidence level honestly
"""
    
    def extract_data(self, image_path, max_retries=3):
        """Extract data with comprehensive error handling"""
        
        try:
            base64_image = self._encode_image(image_path)
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.extraction_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=1500,
                        temperature=0.1
                    )
                    
                    content = response.choices[0].message.content
                    extracted_data = self._parse_json_response(content)
                    
                    if extracted_data:
                        extracted_data['_metadata'] = {
                            'image_path': str(image_path),
                            'extraction_timestamp': datetime.now().isoformat(),
                            'model_version': 'gpt-4o',
                            'attempt_number': attempt + 1
                        }
                        
                        logger.info(f"Successfully extracted data from {image_path}")
                        return extracted_data
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {image_path}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            
            logger.error(f"All extraction attempts failed for {image_path}")
            return None
            
        except Exception as e:
            logger.error(f"Critical error extracting from {image_path}: {e}")
            return None
    
    def _encode_image(self, image_path):
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _parse_json_response(self, content):
        """Parse JSON from GPT-4V response"""
        
        # Try direct JSON parsing
        try:
            return json.loads(content)
        except:
            pass
        
        # Try extracting JSON block
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # Try ```json blocks
        try:
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                if end != -1:
                    json_str = content[start:end].strip()
                    return json.loads(json_str)
        except:
            pass
        
        return None

def maximize_5_dollar_budget():
    """Use complete $5 budget strategically - ~150-170 extractions"""
    
    print("=== MAXIMIZING $5 BUDGET - FULL EXTRACTION ===")
    print("Target: ~150-170 extractions at $0.03 each")
    
    # Check if required files exist
    if not Path('data/chart_configurations.json').exists():
        print("❌ Chart configurations not found. Run the main script first to generate charts.")
        return
    
    # Create extractions folder
    Path('data/extractions').mkdir(exist_ok=True)
    
    # Load chart configurations
    with open('data/chart_configurations.json', 'r') as f:
        all_configs = json.load(f)
    
    print(f"Found {len(all_configs)} total charts")
    
    # PHASE 1: ORIGINAL CHARTS (30 charts = $0.90)
    print("\n=== PHASE 1: ORIGINAL CHARTS ===")
    
    # Strategic selection: diverse sample
    original_sample = []
    
    # Get balanced sample across types and complexities
    chart_types = ['bar', 'pie', 'line', 'stacked_bar', 'grouped_bar', 'area', 'scatter']
    complexities = ['simple', 'medium', 'complex']
    
    for chart_type in chart_types:
        type_charts = [c for c in all_configs if c['type'] == chart_type]
        if type_charts:
            # Get mix of complexities for this type
            for complexity in complexities:
                complexity_type_charts = [c for c in type_charts if c['complexity'] == complexity]
                if complexity_type_charts and len(original_sample) < 30:
                    original_sample.append(random.choice(complexity_type_charts))
    
    # Fill remaining slots randomly
    remaining_charts = [c for c in all_configs if c not in original_sample]
    while len(original_sample) < 30 and remaining_charts:
        original_sample.append(random.choice(remaining_charts))
        remaining_charts.remove(original_sample[-1])
    
    print(f"Selected {len(original_sample)} original charts for extraction")
    
    # Initialize extractor and tracking
    extractor = GPT4VisionExtractor(client)
    results = {}
    total_cost = 0.0
    successful_extractions = 0
    
    # Extract from originals
    for i, config in enumerate(original_sample):
        chart_path = f"data/raw_charts/{config['id']}.png"
        
        if Path(chart_path).exists():
            print(f"ORIGINAL {i+1}/{len(original_sample)}: {config['id']}")
            
            extracted_data = extractor.extract_data(chart_path)
            
            if extracted_data:
                results[config['id']] = {
                    'type': 'original',
                    'extracted': extracted_data,
                    'ground_truth': {
                        'chart_title': config['title'],
                        'chart_type': config['type'],
                        'data': [{'category': cat, 'value': val} 
                               for cat, val in zip(config['categories'], config['values'])]
                    }
                }
                
                # Save individual result
                with open(f"data/extractions/{config['id']}_original.json", 'w') as f:
                    json.dump(extracted_data, f, indent=2)
                
                successful_extractions += 1
                total_cost += 0.03
                print(f"✅ SUCCESS | Total: {successful_extractions} | Cost: ${total_cost:.2f}")
            else:
                print(f"❌ FAILED: {config['id']}")
            
            time.sleep(1)  # Rate limiting
    
    # PHASE 2: PERTURBATIONS (120+ extractions = $3.60+)
    print(f"\n=== PHASE 2: PERTURBATIONS ===")
    print(f"Budget remaining: ${5.00 - total_cost:.2f}")
    
    # Use top 20 original charts for perturbations
    perturbation_base_charts = original_sample[:20]
    
    # Key perturbations in order of importance
    perturbations = [
        'gaussian_blur', 'rotation', 'grayscale', 'brightness', 
        'random_blocks', 'contrast', 'legend_removal'
    ]
    
    # Calculate how many perturbations we can afford
    remaining_budget = 5.00 - total_cost
    max_additional_extractions = int(remaining_budget / 0.03)
    
    print(f"Can afford {max_additional_extractions} more extractions")
    
    # Distribute perturbations across charts
    perturbation_queue = []
    
    for chart in perturbation_base_charts:
        for perturbation in perturbations:
            for intensity in ['medium', 'high', 'low']:  # Prioritize medium first
                pert_path = f"data/perturbations/{chart['id']}_{perturbation}_{intensity}.png"
                if Path(pert_path).exists():
                    perturbation_queue.append({
                        'chart': chart,
                        'perturbation': perturbation,
                        'intensity': intensity,
                        'path': pert_path
                    })
    
    # Shuffle for variety, but prioritize medium intensity
    medium_intensity = [p for p in perturbation_queue if p['intensity'] == 'medium']
    other_intensity = [p for p in perturbation_queue if p['intensity'] != 'medium']
    
    # Process medium first, then others
    ordered_queue = medium_intensity + other_intensity
    
    # Extract perturbations until budget runs out
    perturbation_count = 0
    for item in ordered_queue:
        if total_cost >= 4.95:  # Leave small buffer
            print(f"⚠️ Budget limit reached at ${total_cost:.2f}")
            break
        
        chart = item['chart']
        perturbation = item['perturbation']
        intensity = item['intensity']
        pert_path = item['path']
        
        print(f"PERTURBATION {perturbation_count+1}: {chart['id']}_{perturbation}_{intensity}")
        
        extracted_data = extractor.extract_data(pert_path)
        
        if extracted_data:
            result_key = f"{chart['id']}_{perturbation}_{intensity}"
            results[result_key] = {
                'type': 'perturbation',
                'extracted': extracted_data,
                'original_id': chart['id'],
                'perturbation': perturbation,
                'intensity': intensity
            }
            
            # Save result
            with open(f"data/extractions/{result_key}.json", 'w') as f:
                json.dump(extracted_data, f, indent=2)
            
            successful_extractions += 1
            total_cost += 0.03
            perturbation_count += 1
            print(f"✅ SUCCESS | Total: {successful_extractions} | Cost: ${total_cost:.2f}")
        else:
            print(f"❌ FAILED: {result_key}")
        
        time.sleep(1)
    
    # FINAL SUMMARY
    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Total successful extractions: {successful_extractions}")
    print(f"Original charts: {len([r for r in results.values() if r['type'] == 'original'])}")
    print(f"Perturbed charts: {len([r for r in results.values() if r['type'] == 'perturbation'])}")
    print(f"Final cost: ${total_cost:.2f}")
    print(f"Budget utilization: {(total_cost/5.00)*100:.1f}%")
    
    # Save comprehensive results
    with open('data/full_extraction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    summary = {
        'total_extractions': successful_extractions,
        'original_count': len([r for r in results.values() if r['type'] == 'original']),
        'perturbation_count': len([r for r in results.values() if r['type'] == 'perturbation']),
        'final_cost': round(total_cost, 2),
        'budget_utilization': round((total_cost/5.00)*100, 1),
        'perturbations_tested': list(set([r['perturbation'] for r in results.values() if r['type'] == 'perturbation'])),
        'chart_types_tested': list(set([r['ground_truth']['chart_type'] for r in results.values() if r['type'] == 'original']))
    }
    
    with open('data/extraction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 RESULTS SAVED:")
    print(f"- Full results: data/full_extraction_results.json")
    print(f"- Summary: data/extraction_summary.json")
    print(f"- Individual files: data/extractions/")
    
    return results

if __name__ == "__main__":
    # Check API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        print("❌ Please set your OpenAI API key in .env file first!")
        print("Create .env file with: OPENAI_API_KEY=your-actual-key")
    else:
        print("🚀 MAXIMIZING $5 BUDGET FOR COMPLETE EXTRACTION")
        print(f"API Key loaded: {OPENAI_API_KEY[:8]}...")
        
        # Confirm budget usage
        confirm = input("\n⚠️ This will use your COMPLETE $5 budget (~150+ extractions). Continue? (y/n): ")
        
        if confirm.lower() == 'y':
            print("🎯 Starting maximum extraction...")
            results = maximize_5_dollar_budget()
            
            if results:
                print("✅ MAXIMUM EXTRACTION COMPLETED!")
                print("🎓 You now have enough data for a complete dissertation!")
            else:
                print("❌ Extraction failed!")
        else:
            print("⏸️ Extraction cancelled.")