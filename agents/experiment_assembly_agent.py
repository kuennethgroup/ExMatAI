"""
Experiment Assembly Agent - Match materials with SMILES and performance data
"""

from typing import Dict, List
from utils.state_schema import AgentState

def experiment_assembly_agent(state: AgentState) -> AgentState:
    """
    Assemble complete experiments by matching:
    - Battery stacks (cathode + anode + electrolyte)
    - Material SMILES
    - Performance data (cycling + voltage)
    """
    print("\n" + "="*80)
    print("🔧 AGENT 6: Experiment Assembly & Data Matching")
    print("="*80)
    
    battery_stacks = state.get('battery_stacks', [])
    material_smiles = state.get('material_smiles', {})
    cycling_data = state.get('cycling_data', [])
    voltage_data = state.get('voltage_data', [])
    identified_materials = state.get('identified_materials', [])
    processing_params = state.get('processing_params', {})
    
    print(f"\n📊 Assembly summary:")
    print(f"  ├─ Battery stacks: {len(battery_stacks)}")
    print(f"  ├─ Materials with SMILES: {len(material_smiles)}")
    print(f"  ├─ Cycling datasets: {len(cycling_data)}")
    print(f"  └─ Voltage datasets: {len(voltage_data)}")
    
    experiments = []
    
    if not battery_stacks:
        print("\n⚠️  No battery stacks to assemble experiments")
        
        # Create fallback experiment from identified materials
        if identified_materials:
            print(f"  ├─ Creating experiment from {len(identified_materials)} identified materials...")
            
            # Group materials by role
            cathodes = [m for m in identified_materials if m.get('role') == 'cathode']
            anodes = [m for m in identified_materials if m.get('role') == 'anode']
            electrolytes = [m for m in identified_materials if m.get('role') == 'electrolyte']
            
            experiment = {
                'experiment_id': 1,
                'materials': {
                    'cathode': cathodes[0] if cathodes else {'name': 'Unknown', 'formula': 'Unknown'},
                    'anode': anodes[0] if anodes else {'name': 'Unknown', 'formula': 'Unknown'},
                    'electrolyte': electrolytes if electrolytes else [{'name': 'Unknown', 'formula': 'Unknown'}]
                },
                'smiles': {},
                'processing_conditions': processing_params,
                'performance_data': {
                    'cycling': cycling_data[0] if cycling_data else None,
                    'voltage_profile': voltage_data[0] if voltage_data else None
                }
            }
            
            # Match SMILES to materials
            for material_name, smiles_info in material_smiles.items():
                # Try to match by name similarity
                for material in identified_materials:
                    mat_name = material.get('material_name', '').lower()
                    if material_name.lower() in mat_name or mat_name in material_name.lower():
                        experiment['smiles'][mat_name] = smiles_info['smiles']
                        break
            
            experiments.append(experiment)
            print(f"  ✓ Created fallback experiment")
    
    else:
        # Create experiments from battery stacks
        print(f"\n🔨 Assembling {len(battery_stacks)} experiments...")
        
        for stack_idx, stack in enumerate(battery_stacks, 1):
            print(f"\n  ├─ Experiment {stack_idx}:")
            
            experiment = {
                'experiment_id': stack_idx,
                'materials': {
                    'cathode': stack['cathode'],
                    'anode': stack['anode'],
                    'electrolyte': stack['electrolyte']
                },
                'smiles': {},
                'processing_conditions': stack.get('processing_conditions', {}),
                'performance_data': {
                    'cycling': None,
                    'voltage_profile': None
                }
            }
            
            # Match SMILES to materials in this stack
            print(f"  │  ├─ Matching SMILES...")
            matched_count = 0
            
            # Try to match cathode
            cathode_name = stack['cathode']['name'].lower()
            for mat_name, smiles_info in material_smiles.items():
                if mat_name.lower() in cathode_name or cathode_name in mat_name.lower():
                    experiment['smiles']['cathode'] = smiles_info['smiles']
                    matched_count += 1
                    print(f"  │  │  ✓ Matched cathode SMILES")
                    break
            
            # Try to match anode
            anode_name = stack['anode']['name'].lower()
            for mat_name, smiles_info in material_smiles.items():
                if mat_name.lower() in anode_name or anode_name in mat_name.lower():
                    experiment['smiles']['anode'] = smiles_info['smiles']
                    matched_count += 1
                    print(f"  │  │  ✓ Matched anode SMILES")
                    break
            
            print(f"  │  └─ Matched {matched_count} SMILES")
            
            # Attach performance data (distribute evenly if multiple stacks)
            if cycling_data and stack_idx <= len(cycling_data):
                experiment['performance_data']['cycling'] = cycling_data[stack_idx - 1]
                print(f"  │  ✓ Attached cycling data ({cycling_data[stack_idx - 1]['data_points']} points)")
            
            if voltage_data and stack_idx <= len(voltage_data):
                experiment['performance_data']['voltage_profile'] = voltage_data[stack_idx - 1]
                print(f"  │  ✓ Attached voltage data ({voltage_data[stack_idx - 1]['data_points']} points)")
            
            experiments.append(experiment)
    
    print(f"\n" + "="*80)
    print(f"✅ Experiment Assembly Complete:")
    print(f"  ├─ Total experiments: {len(experiments)}")
    
    for exp in experiments:
        smiles_count = len(exp['smiles'])
        has_cycling = exp['performance_data']['cycling'] is not None
        has_voltage = exp['performance_data']['voltage_profile'] is not None
        print(f"  │  • Exp {exp['experiment_id']}: {smiles_count} SMILES, " +
              f"cycling={'✓' if has_cycling else '✗'}, voltage={'✓' if has_voltage else '✗'}")
    
    print("="*80)
    
    return {
        **state,
        "experiments": experiments,
        "messages": state.get('messages', []) + ["Experiment assembly completed"],
        "current_agent": "data_aggregation"
    }
