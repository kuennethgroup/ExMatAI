"""
Validation utilities
"""

from rdkit import Chem
from typing import Dict, List, Optional
import re

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def validate_battery_stack(cathode: Dict, anode: Dict, electrolyte: Dict) -> tuple[bool, List[str]]:
    """Validate battery stack has all required components"""
    errors = []
    
    if not cathode or not cathode.get('material_name'):
        errors.append("Missing cathode material")
    
    if not anode or not anode.get('material_name'):
        errors.append("Missing anode material")
    
    if not electrolyte or not electrolyte.get('components'):
        errors.append("Missing electrolyte components")
    
    return len(errors) == 0, errors

def validate_experiment(experiment: Dict) -> tuple[bool, List[str]]:
    """Validate complete experiment record"""
    errors = []
    
    required_fields = ['experiment_id', 'type', 'cathode', 'anode', 'electrolyte']
    
    for field in required_fields:
        if field not in experiment:
            errors.append(f"Missing required field: {field}")
    
    # Validate experiment type
    if experiment.get('type') not in ['Half-cell', 'Full-cell']:
        errors.append(f"Invalid experiment type: {experiment.get('type')}")
    
    # Validate subtype for half-cell
    if experiment.get('type') == 'Half-cell':
        if experiment.get('subtype') not in ['Positive', 'Negative']:
            errors.append(f"Invalid half-cell subtype: {experiment.get('subtype')}")
    
    return len(errors) == 0, errors

def clean_numeric_value(value_str: str) -> Optional[float]:
    """Extract numeric value from string"""
    if not value_str:
        return None
    
    # Remove units and extra text
    match = re.search(r'(\d+\.?\d*)', str(value_str))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    return None
