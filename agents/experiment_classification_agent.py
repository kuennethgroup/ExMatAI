"""
Experiment Classification Agent - Classify and structure experiments
"""

from typing import Dict, List, Optional
from utils.state_schema import AgentState

def experiment_classification_agent(state: AgentState) -> AgentState:
    """
    Classify experiments as half-cell or full-cell and create structured records
    """
    print("\n" + "="*80)
    print("🔬 AGENT 6: Experiment Classification & Assembly")
    print("="*80)
    
    experiments = []
    experiment_id = 1
    
    for stack_idx, stack in enumerate(state['battery_stacks'], 1):
        print(f"\n📦 Processing battery stack {stack_idx}/{len(state['battery_stacks'])}...")
        print(f"  ├─ Cathode: {stack['cathode']}")
        print(f"  ├─ Anode: {stack['anode']}")
        print(f"  └─ Electrolyte: {', '.join(stack['electrolyte'])}")
        
        try:
            # Get material info
            cathode_name = stack['cathode']
            anode_name = stack['anode']
            electrolyte_names = stack['electrolyte']
            
            # Determine experiment type
            is_cathode_reference = cathode_name.lower() in ['[pt]', 'pt', 'platinum']
            is_anode_reference = anode_name.lower() in ['[pt]', 'pt', 'platinum']
            
            if is_cathode_reference or is_anode_reference:
                experiment_type = "Half-cell"
                subtype = "Positive" if is_anode_reference else "Negative"
            else:
                experiment_type = "Full-cell"
                subtype = None
            
            print(f"\n  📋 Experiment Type: {experiment_type}" + (f" ({subtype})" if subtype else ""))
            
            # Build cathode composition
            cathode_info = build_material_composition(
                cathode_name,
                state['identified_materials'],
                state['material_smiles'],
                'cathode'
            )
            
            # Build anode composition
            anode_info = build_material_composition(
                anode_name,
                state['identified_materials'],
                state['material_smiles'],
                'anode'
            )
            
            # Build electrolyte composition
            electrolyte_info = build_electrolyte_composition(
                electrolyte_names,
                state['identified_materials'],
                state['material_smiles']
            )
            
            # Match performance data
            performance_data = match_performance_data(
                stack,
                experiment_type,
                subtype,
                state['cycling_data'],
                state['voltage_data']
            )
            
            # Create experiment record
            experiment = {
                'experiment_id': experiment_id,
                'type': experiment_type,
                'subtype': subtype,
                'cathode': cathode_info,
                'anode': anode_info,
                'electrolyte': electrolyte_info,
                'processing_parameters': state['processing_params'],
                'reference_electrode': state['processing_params'].get('reference_electrode'),
                'cell_setup': state['processing_params'].get('cell_setup', ''),
                'performance_data': performance_data
            }
            
            experiments.append(experiment)
            print(f"  ✓ Created experiment {experiment_id}")
            experiment_id += 1
        
        except Exception as e:
            print(f"  ✗ Failed to create experiment: {e}")
            state['errors'].append(f"Experiment creation failed for stack {stack_idx}: {e}")
    
    print(f"\n" + "="*80)
    print(f"✅ Experiment Classification Complete:")
    print(f"  └─ Total experiments: {len(experiments)}")
    print("="*80)
    
    return {
        **state,
        "experiments": experiments,
        "messages": state.get('messages', []) + ["Experiment classification completed"],
        "current_agent": "data_aggregation"
    }

def build_material_composition(
    material_name: str,
    identified_materials: List[Dict],
    material_smiles: Dict,
    role: str
) -> Dict:
    """Build complete material composition with SMILES"""
    
    # Find material info
    material = next(
        (m for m in identified_materials 
         if (m.get('abbreviation') or m.get('full_name')) == material_name),
        None
    )
    
    if not material:
        # Handle reference electrodes
        return {
            'material_name': material_name,
            'full_name': None,
            'abbreviation': None,
            'smiles': None,
            'wt_percent': None,
            'confidence': 1.0,
            'source_page': None,
            'role': 'reference_electrode'
        }
    
    # Get SMILES if available
    smiles_info = material_smiles.get(material_name, {})
    
    return {
        'material_name': material_name,
        'full_name': material.get('full_name'),
        'abbreviation': material.get('abbreviation'),
        'smiles': smiles_info.get('smiles'),
        'wt_percent': material.get('wt_percent'),
        'binder': material.get('binder'),
        'binder_wt_percent': material.get('binder_wt_percent'),
        'conductive_material': material.get('conductive_material'),
        'conductive_wt_percent': material.get('conductive_wt_percent'),
        'confidence': smiles_info.get('confidence', 0.0),
        'source_page': smiles_info.get('source_page')
    }

def build_electrolyte_composition(
    electrolyte_names: List[str],
    identified_materials: List[Dict],
    material_smiles: Dict
) -> Dict:
    """Build electrolyte composition with components"""
    
    components = []
    description_parts = []
    
    for name in electrolyte_names:
        material = next(
            (m for m in identified_materials 
             if (m.get('abbreviation') or m.get('full_name')) == name),
            None
        )
        
        if not material:
            continue
        
        smiles_info = material_smiles.get(name, {})
        
        # Determine component type
        comp_type = 'salt' if 'salt' in material['role'] else 'solvent'
        
        component = {
            'type': comp_type,
            'name': name,
            'smiles': smiles_info.get('smiles'),
            'concentration': material.get('concentration'),
            'confidence': smiles_info.get('confidence', 0.0),
            'source_page': smiles_info.get('source_page')
        }
        
        components.append(component)
        
        # Build description
        if material.get('mentions'):
            for mention in material['mentions']:
                if 'M' in mention or 'mmol' in mention:
                    description_parts.append(mention)
                    break
    
    description = ' in '.join(description_parts) if description_parts else ', '.join(electrolyte_names)
    
    return {
        'description': description,
        'components': components
    }

def match_performance_data(
    stack: Dict,
    exp_type: str,
    subtype: Optional[str],
    cycling_data: List[Dict],
    voltage_data: List[Dict]
) -> Dict:
    """Match performance data to experiment"""
    
    # Simple matching based on caption content
    # In production, this should be more sophisticated
    
    matched_cycling = []
    matched_voltage = []
    
    search_terms = [
        stack['cathode'].lower(),
        stack['anode'].lower()
    ]
    
    for cycle_dataset in cycling_data:
        caption = cycle_dataset.get('caption', '').lower()
        if any(term in caption for term in search_terms):
            matched_cycling.append(cycle_dataset)
    
    for voltage_dataset in voltage_data:
        caption = voltage_dataset.get('caption', '').lower()
        if any(term in caption for term in search_terms):
            matched_voltage.append(voltage_dataset)
    
    # If no match, include all data (fallback)
    if not matched_cycling:
        matched_cycling = cycling_data
    if not matched_voltage:
        matched_voltage = voltage_data
    
    return {
        'specific_capacity_vs_cycles': matched_cycling,
        'voltage_profiles': matched_voltage
    }
