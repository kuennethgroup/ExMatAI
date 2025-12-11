"""
SMILES Generation Agent - Convert structures to SMILES using ChemVLM
"""

import torch
from rdkit import Chem
from typing import Dict, List
from utils.state_schema import AgentState
from utils.chemvlm_wrapper import ChemVLMWrapper

def smiles_generation_agent(state: AgentState) -> AgentState:
    """
    Generate SMILES from structure images using ChemVLM
    """
    print("\n" + "="*80)
    print("🧪 AGENT 4: SMILES Generation (ChemVLM)")
    print("="*80)
    
    detected_structures = state.get('detected_structures', [])
    
    if not detected_structures:
        print("\n⚠️  No detected structures - skipping SMILES generation")
        return {
            **state,
            "material_smiles": {},
            "messages": state.get('messages', []) + ["No structures for SMILES generation"],
            "current_agent": "plots_analysis"
        }
    
    print(f"\n🔬 Generating SMILES for {len(detected_structures)} structures...")
    
    # Initialize ChemVLM
    try:
        chemvlm = ChemVLMWrapper()
        
        if not chemvlm.load_model():
            print(f"  ✗ ChemVLM not available")
            return {
                **state,
                "material_smiles": {},
                "messages": state.get('messages', []) + ["ChemVLM not available"],
                "current_agent": "plots_analysis"
            }
        
    except Exception as e:
        print(f"  ✗ Failed to initialize ChemVLM: {e}")
        return {
            **state,
            "material_smiles": {},
            "errors": state.get('errors', []) + [f"ChemVLM init failed: {e}"],
            "messages": state.get('messages', []) + ["SMILES generation failed"],
            "current_agent": "plots_analysis"
        }
    
    material_smiles = {}
    
    for idx, structure in enumerate(detected_structures, 1):
        print(f"\n  ├─ Processing structure {idx}/{len(detected_structures)}: '{structure['material_name']}'...")
        
        try:
            # Generate SMILES using ChemVLM
            smiles = chemvlm.generate_smiles(structure['image'])
            
            # Clean SMILES (remove any explanatory text)
            smiles = smiles.split('\n')[0].strip()
            
            # Validate SMILES with RDKit
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                # Canonicalize SMILES
                canonical_smiles = Chem.MolToSmiles(mol)
                
                print(f"  │  ✓ Valid SMILES: {canonical_smiles}")
                
                material_name = structure.get('material_name', f'structure_{idx}')
                material_smiles[material_name] = {
                    'smiles': canonical_smiles,
                    'page': structure['page'],
                    'confidence': structure.get('confidence', 0.0),
                    'bbox': structure['bbox']
                }
            else:
                print(f"  │  ⚠️  Generated invalid SMILES: {smiles}")
                state['errors'].append(f"Invalid SMILES for {structure.get('material_name', 'unknown')}")
                
        except Exception as e:
            print(f"  │  ✗ SMILES generation failed: {e}")
            state['errors'].append(f"SMILES generation failed for structure {idx}: {e}")
    
    print(f"\n" + "="*80)
    print(f"✅ SMILES Generation Complete:")
    print(f"  ├─ Structures processed: {len(detected_structures)}")
    print(f"  └─ Valid SMILES generated: {len(material_smiles)}")
    print("="*80)
    
    return {
        **state,
        "material_smiles": material_smiles,
        "messages": state.get('messages', []) + ["SMILES generation completed"],
        "current_agent": "plots_analysis"
    }
