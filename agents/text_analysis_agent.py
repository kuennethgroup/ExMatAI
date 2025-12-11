"""
Text Analysis Agent - Identify materials and validate battery components
"""

import json
import ollama
import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from utils.state_schema import AgentState

# Configure Ollama client
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
client = ollama.Client(host=OLLAMA_HOST)

def text_analysis_agent(state: AgentState) -> AgentState:
    """
    Analyze text to identify materials, validate battery components
    """
    print("\n" + "="*80)
    print("📝 AGENT 2: Text Analysis & Material Identification")
    print("="*80)
    
    extracted_text = state['extracted_text']
    
    # 1. Extract paper metadata
    print("\n📊 Extracting paper metadata...")
    metadata_prompt = f"""
Extract paper metadata from this battery research paper text. Return ONLY valid JSON with these exact fields:
- doi (string)
- title (string)
- authors (array of strings)
- journal (string)
- year (integer)

Text (first 3000 chars):
{extracted_text[:3000]}

Return ONLY the JSON object, no other text or explanation.
"""
    
    try:
        metadata_response = client.generate(
            model='qwen3:8b',  # TEXT MODEL
            prompt=metadata_prompt,
            format='json'
        )
        
        response_text = metadata_response['response']
        paper_info = json.loads(response_text) if isinstance(response_text, str) else response_text
        
        print(f"  ✓ Paper: {paper_info.get('title', 'Unknown')[:60]}...")
        
    except Exception as e:
        print(f"  ✗ Metadata extraction failed: {e}")
        paper_info = {
            "doi": "unknown",
            "title": extracted_text[:100] if extracted_text else "Unknown",
            "authors": [],
            "journal": "Unknown",
            "year": 2024
        }
    
    # 2. Identify battery materials with roles
    print("\n🔬 Identifying battery materials...")
    materials_prompt = f"""
Analyze this battery research paper and identify ALL battery materials mentioned.

For EACH material, extract:
- material_name: Full name (e.g., "Prussian blue", "PTCDI", "KCl")
- role: Must be one of: "cathode", "anode", "electrolyte", or "other"
- chemical_formula: If mentioned (e.g., "KFe[Fe(CN)6]", "C24H8N2O8")

Text:
{extracted_text[:8000]}

Return as JSON array. Example:
[
  {{"material_name": "Prussian blue", "role": "cathode", "chemical_formula": "KFe[Fe(CN)6]"}},
  {{"material_name": "PTCDI", "role": "anode", "chemical_formula": "C24H8N2O8"}},
  {{"material_name": "KCl", "role": "electrolyte", "chemical_formula": "KCl"}}
]

Return ONLY the JSON array, no other text.
"""
    
    try:
        materials_response = client.generate(
            model='qwen3:8b',  # TEXT MODEL
            prompt=materials_prompt,
            format='json'
        )
        
        response_text = materials_response['response']
        identified_materials = json.loads(response_text) if isinstance(response_text, str) else response_text
        
        if not isinstance(identified_materials, list):
            identified_materials = []
        
        print(f"  ✓ Found {len(identified_materials)} materials:")
        for mat in identified_materials[:5]:
            if isinstance(mat, dict):
                print(f"    • {mat.get('material_name', 'Unknown')} ({mat.get('role', 'unknown')})")
        
    except Exception as e:
        print(f"  ✗ Material identification failed: {e}")
        identified_materials = []
    
    # 3. Extract composition and processing details
    print("\n⚗️  Extracting composition & processing details...")
    details_prompt = f"""
Extract battery material composition and processing details from this text.

Find:
- mass_loading_mg_cm2 (number)
- active_material_ratio (percentage as number, e.g., 70 for 70%)
- current_density_mA_g (number)
- voltage_range_V (string like "1.2-3.9")
- temperature_C (number, default 25 if room temp mentioned)
- cycle_life (number of cycles tested)

Text:
{extracted_text[:6000]}

Return as JSON object. Example:
{{
  "mass_loading_mg_cm2": 1.5,
  "active_material_ratio": 70,
  "current_density_mA_g": 400,
  "voltage_range_V": "1.2-3.9",
  "temperature_C": 25,
  "cycle_life": 10000
}}

Return ONLY JSON object.
"""
    
    try:
        details_response = client.generate(
            model='qwen3:8b',
            prompt=details_prompt,
            format='json'
        )
        
        response_text = details_response['response']
        processing_params = json.loads(response_text) if isinstance(response_text, str) else response_text
        
        if not isinstance(processing_params, dict):
            processing_params = {}
        
        print(f"  ✓ Extracted {len(processing_params)} parameters")
        
    except Exception as e:
        print(f"  ⚠️  Details extraction had issues: {e}")
        processing_params = {}
    
    # 4. Build battery stacks
    print("\n✓ Building battery stacks...")
    
    cathodes = [m for m in identified_materials if isinstance(m, dict) and m.get('role') == 'cathode']
    anodes = [m for m in identified_materials if isinstance(m, dict) and m.get('role') == 'anode']
    electrolytes = [m for m in identified_materials if isinstance(m, dict) and m.get('role') == 'electrolyte']
    
    battery_stacks = []
    errors = []
    
    if cathodes and anodes:
        for cathode in cathodes:
            for anode in anodes:
                stack = {
                    'cathode': {
                        'name': cathode.get('material_name', 'Unknown'),
                        'formula': cathode.get('chemical_formula', 'Unknown')
                    },
                    'anode': {
                        'name': anode.get('material_name', 'Unknown'),
                        'formula': anode.get('chemical_formula', 'Unknown')
                    },
                    'electrolyte': [
                        {
                            'name': e.get('material_name', 'Unknown'),
                            'formula': e.get('chemical_formula', 'Unknown')
                        } for e in electrolytes
                    ] if electrolytes else [{'name': 'Unknown', 'formula': 'Unknown'}],
                    'processing_conditions': processing_params
                }
                battery_stacks.append(stack)
        
        print(f"  ✓ Created {len(battery_stacks)} battery stack(s)")
    else:
        if not cathodes:
            errors.append("No cathode materials found")
        if not anodes:
            errors.append("No anode materials found")
        print(f"  ⚠️  Could not create complete stacks")
    
    print("="*80)
    
    return {
        **state,
        "paper_info": paper_info,
        "identified_materials": identified_materials,
        "processing_params": processing_params,
        "battery_stacks": battery_stacks,
        "errors": state.get('errors', []) + errors,
        "messages": state.get('messages', []) + ["Text analysis completed"],
        "current_agent": "structure_extraction"
    }
