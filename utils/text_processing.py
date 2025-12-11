"""
Text processing utilities
"""

import re
from typing import List, Dict, Optional
from difflib import get_close_matches

def extract_structure_labels(page_text: str, analysis: str) -> List[str]:
    """Extract potential structure labels from text"""
    labels = []
    
    # Common patterns for chemical names/abbreviations
    patterns = [
        r'\b[A-Z][A-Za-z]*[A-Z]\w*\b',  # PTPAn, PTCDI, KFSI
        r'\b(?:Figure|Fig\.?)\s*\d+[a-z]?\b',  # Figure references
        r'\b(?:Scheme|Chart)\s*\d+\b',  # Scheme references
        r'\b[A-Z]{2,}\b',  # Abbreviations
        r'\b(?:compound|molecule|material)\s+(\w+)\b',  # Direct references
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        labels.extend(matches)
    
    # Extract from analysis text
    if 'label' in analysis.lower():
        label_matches = re.findall(r'label[ed]*\s+["\']?(\w+)["\']?', analysis, re.IGNORECASE)
        labels.extend(label_matches)
    
    # Clean and deduplicate
    labels = [label.strip() for label in labels if len(label) > 1]
    return list(set(labels))

def extract_figure_caption(page_text: str, page_num: int) -> str:
    """Extract figure caption from page text"""
    pattern = r'(?:Figure|Fig\.?)\s*(\d+[a-z]?)[:\.]?\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)'
    
    matches = re.finditer(pattern, page_text, re.IGNORECASE | re.MULTILINE)
    
    for match in matches:
        fig_num = match.group(1)
        caption = match.group(2).strip()
        return f"Figure {fig_num}: {caption}"
    
    return f"Figure on page {page_num}"

def extract_figure_reference_structured(caption: str, page: int) -> Dict:
    """Extract structured figure reference"""
    pattern = r'(?:Figure|Fig\.?)\s*(\d+)([a-z]?)'
    match = re.search(pattern, caption, re.IGNORECASE)
    
    if match:
        return {
            'figure': int(match.group(1)),
            'subplot': match.group(2) if match.group(2) else None,
            'page': page
        }
    
    return {'figure': None, 'subplot': None, 'page': page}

def match_structure_label(nearby_text: List[str], material_terms: Dict[str, str]) -> Optional[str]:
    """Match structure to material name using nearby text"""
    text_combined = " ".join(nearby_text).lower()
    
    # Direct match
    for term_lower, material_name in material_terms.items():
        if term_lower in text_combined:
            return material_name
    
    # Fuzzy match
    for text in nearby_text:
        text_clean = text.strip().lower()
        matches = get_close_matches(text_clean, material_terms.keys(), n=1, cutoff=0.8)
        if matches:
            return material_terms[matches[0]]
    
    return None

def clean_smiles_response(response: str) -> str:
    """Extract clean SMILES from model response"""
    # Remove common prefixes
    response = response.strip()
    
    # Look for SMILES pattern
    smiles_pattern = r'([A-Za-z0-9@+\-\[\]\(\)=#$:/\\]+)'
    match = re.search(smiles_pattern, response)
    
    if match:
        return match.group(1)
    
    return response
