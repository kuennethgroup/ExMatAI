"""
Plots Analysis Agent - Extract performance data from plots using Qwen3-VL
"""

import os
import json
import ollama
from PIL import Image
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from utils.state_schema import AgentState

# Configure Ollama client
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
client = ollama.Client(host=OLLAMA_HOST)

def plots_analysis_agent(state: AgentState) -> AgentState:
    """
    Extract performance data from cycling and voltage plots
    """
    print("\n" + "="*80)
    print("📊 AGENT 5: Plot Data Extraction (Qwen3-VL)")
    print("="*80)
    
    plot_images = state.get('plot_images', [])
    
    if not plot_images:
        print("\n⚠️  No plot images to analyze - skipping")
        return {
            **state,
            "cycling_data": [],
            "voltage_data": [],
            "messages": state.get('messages', []) + ["No plots to analyze"],
            "current_agent": "experiment_assembly"
        }
    
    print(f"\n📈 Analyzing {len(plot_images)} plots...")
    
    cycling_data = []
    voltage_data = []
    
    for idx, plot_img in enumerate(plot_images, 1):
        print(f"\n  ├─ Analyzing plot {idx}/{len(plot_images)} (page {plot_img['page']})...")
        
        try:
            # Save plot temporarily
            temp_path = f"temp/plot_p{plot_img['page']}_{idx}.png"
            plot_img['image'].save(temp_path)
            
            # Step 1: Classify plot type
            classification_prompt = """
Analyze this plot and determine its type. Return ONLY ONE of these categories:
- "capacity_vs_cycle" (capacity retention over cycle number)
- "voltage_profile" (voltage vs capacity/time)
- "rate_capability" (capacity at different rates)
- "other"

Return only the category name, nothing else.
"""
            
            classification = client.generate(
                model='qwen3-vl:8b',
                prompt=classification_prompt,
                images=[temp_path]
            )
            
            plot_type = classification['response'].strip().lower()
            print(f"  │  ├─ Plot type: {plot_type}")
            
            # Step 2: Extract data based on plot type
            if 'capacity' in plot_type and 'cycle' in plot_type:
                # Extract cycling data
                data_prompt = """
Extract ALL data points from this capacity vs cycle number plot.

Return as JSON array with this format:
[
  {"cycle": 1, "capacity_mAh_g": 395},
  {"cycle": 10, "capacity_mAh_g": 390},
  {"cycle": 50, "capacity_mAh_g": 380},
  ...
]

Extract AS MANY data points as visible in the plot. Return ONLY the JSON array.
"""
                
                data_response = client.generate(
                    model='qwen3-vl:8b',
                    prompt=data_prompt,
                    format='json',
                    images=[temp_path]
                )
                
                response_text = data_response['response']
                extracted_data = json.loads(response_text) if isinstance(response_text, str) else response_text
                
                if isinstance(extracted_data, list) and len(extracted_data) > 0:
                    cycling_data.append({
                        'page': plot_img['page'],
                        'caption': plot_img.get('caption', ''),
                        'data': extracted_data,
                        'data_points': len(extracted_data)
                    })
                    print(f"  │  ✓ Extracted {len(extracted_data)} cycling data points")
                else:
                    print(f"  │  ⚠️  No cycling data extracted")
            
            elif 'voltage' in plot_type:
                # Extract voltage profile data
                data_prompt = """
Extract ALL data points from this voltage profile plot.

Return as JSON array with this format:
[
  {"capacity_mAh_g": 0, "voltage_V": 3.5},
  {"capacity_mAh_g": 50, "voltage_V": 3.3},
  {"capacity_mAh_g": 100, "voltage_V": 3.0},
  ...
]

Extract AS MANY data points as visible. Return ONLY the JSON array.
"""
                
                data_response = client.generate(
                    model='qwen3-vl:8b',
                    prompt=data_prompt,
                    format='json',
                    images=[temp_path]
                )
                
                response_text = data_response['response']
                extracted_data = json.loads(response_text) if isinstance(response_text, str) else response_text
                
                if isinstance(extracted_data, list) and len(extracted_data) > 0:
                    voltage_data.append({
                        'page': plot_img['page'],
                        'caption': plot_img.get('caption', ''),
                        'data': extracted_data,
                        'data_points': len(extracted_data)
                    })
                    print(f"  │  ✓ Extracted {len(extracted_data)} voltage data points")
                else:
                    print(f"  │  ⚠️  No voltage data extracted")
            
            elif 'rate' in plot_type:
                # Extract rate capability data
                data_prompt = """
Extract rate capability data from this plot.

Return as JSON array with this format:
[
  {"rate": "1C", "capacity_mAh_g": 395},
  {"rate": "2C", "capacity_mAh_g": 334},
  {"rate": "5C", "capacity_mAh_g": 280},
  ...
]

Extract ALL visible data points. Return ONLY the JSON array.
"""
                
                data_response = client.generate(
                    model='qwen3-vl:8b',
                    prompt=data_prompt,
                    format='json',
                    images=[temp_path]
                )
                
                response_text = data_response['response']
                extracted_data = json.loads(response_text) if isinstance(response_text, str) else response_text
                
                if isinstance(extracted_data, list) and len(extracted_data) > 0:
                    cycling_data.append({
                        'page': plot_img['page'],
                        'caption': plot_img.get('caption', ''),
                        'data': extracted_data,
                        'data_points': len(extracted_data),
                        'data_type': 'rate_capability'
                    })
                    print(f"  │  ✓ Extracted {len(extracted_data)} rate capability points")
                else:
                    print(f"  │  ⚠️  No rate data extracted")
            
            else:
                print(f"  │  ⚠️  Skipping plot type: {plot_type}")
        
        except Exception as e:
            print(f"  │  ✗ Plot analysis failed: {e}")
            state['errors'].append(f"Plot analysis failed for page {plot_img['page']}: {e}")
    
    print(f"\n" + "="*80)
    print(f"✅ Plot Analysis Complete:")
    print(f"  ├─ Plots analyzed: {len(plot_images)}")
    print(f"  ├─ Cycling datasets: {len(cycling_data)}")
    print(f"  └─ Voltage datasets: {len(voltage_data)}")
    print("="*80)
    
    return {
        **state,
        "cycling_data": cycling_data,
        "voltage_data": voltage_data,
        "messages": state.get('messages', []) + ["Plot analysis completed"],
        "current_agent": "experiment_assembly"
    }
