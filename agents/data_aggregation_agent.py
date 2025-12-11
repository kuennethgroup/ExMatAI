"""
Data Aggregation Agent - Compile all extracted data into final JSON
"""

import json
import os
from datetime import datetime
from typing import Dict
from utils.state_schema import AgentState

def data_aggregation_agent(state: AgentState) -> AgentState:
    """
    Aggregate all extracted data and save to structured JSON
    """
    print("\n" + "="*80)
    print("📦 AGENT 7: Data Aggregation & JSON Export")
    print("="*80)
    
    # Prepare comprehensive output data structure
    output_data = {
        "paper_info": state.get('paper_info', {}),
        "extraction_metadata": {
            "extraction_date": datetime.now().isoformat(),
            "pdf_path": state.get('pdf_path', ''),
            "models_used": {
                "ocr": "deepseek-ocr",
                "text_analysis": "qwen3:8b",
                "structure_detection": "UniParser/MolDetv2",
                "structure_labeling": "qwen3-vl:8b",
                "smiles_generation": "AI4Chem/ChemVLM-26B-1-2",
                "plots_analysis": "qwen3-vl:8b"
            },
            "total_experiments": len(state.get('experiments', [])),
            "total_materials": len(state.get('identified_materials', [])),
            "total_structures_detected": len(state.get('detected_structures', [])),
            "total_smiles_generated": len(state.get('material_smiles', {})),
            "errors": state.get('errors', [])
        },
        "materials": {
            "identified_materials": state.get('identified_materials', []),
            "material_smiles": state.get('material_smiles', {}),
            "detected_structures": [
                {
                    'material_name': s.get('material_name'),
                    'page': s.get('page'),
                    'confidence': s.get('confidence'),
                    'bbox': s.get('bbox')
                } for s in state.get('detected_structures', [])
            ]
        },
        "experiments": state.get('experiments', [])
    }
    
    # Generate output filename
    pdf_name = os.path.basename(state['pdf_path']).replace('.pdf', '')
    output_file = f"outputs/{pdf_name}_extracted.json"
    
    # Save to JSON
    os.makedirs("outputs", exist_ok=True)
    
    print(f"\n💾 Saving output to: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved successfully")
        
    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        state['errors'].append(f"Failed to save output: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ Data Aggregation Complete!")
    print(f"{'='*80}")
    
    # Print detailed summary
    print(f"\n📊 Extraction Summary:")
    print(f"\n  Paper Information:")
    paper = output_data['paper_info']
    print(f"    ├─ Title: {paper.get('title', 'Unknown')[:70]}...")
    print(f"    ├─ Authors: {len(paper.get('authors', []))} author(s)")
    print(f"    ├─ Journal: {paper.get('journal', 'Unknown')}")
    print(f"    └─ Year: {paper.get('year', 'Unknown')}")
    
    print(f"\n  Materials:")
    print(f"    ├─ Identified: {len(state.get('identified_materials', []))}")
    print(f"    ├─ Structures detected: {len(state.get('detected_structures', []))}")
    print(f"    └─ SMILES generated: {len(state.get('material_smiles', {}))}")
    
    print(f"\n  Experiments:")
    print(f"    └─ Total: {len(state.get('experiments', []))}")
    
    for exp in state.get('experiments', [])[:3]:  # Show first 3
        print(f"         • Exp {exp['experiment_id']}:")
        print(f"           ├─ Cathode: {exp['materials']['cathode']['name']}")
        print(f"           ├─ Anode: {exp['materials']['anode']['name']}")
        print(f"           ├─ SMILES: {len(exp['smiles'])}")
        cycling = exp['performance_data'].get('cycling')
        if cycling:
            print(f"           └─ Cycling data: {cycling.get('data_points', 0)} points")
    
    print(f"\n  Performance Data:")
    print(f"    ├─ Cycling datasets: {len(state.get('cycling_data', []))}")
    print(f"    └─ Voltage datasets: {len(state.get('voltage_data', []))}")
    
    if state.get('errors'):
        print(f"\n  ⚠️  Errors: {len(state.get('errors', []))}")
        for error in state.get('errors', [])[:3]:  # Show first 3
            print(f"    • {error}")
    
    print(f"\n📄 Output file: {output_file}")
    print("="*80)
    
    return {
        **state,
        "output_file": output_file,
        "messages": state.get('messages', []) + ["Data aggregation completed"],
        "current_agent": "end"
    }
