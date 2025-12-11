"""
Unit tests for individual agents
"""

import pytest
import os
from unittest.mock import Mock, patch

from agents.ocr_agent import ocr_agent
from agents.text_analysis_agent import text_analysis_agent
from utils.state_schema import AgentState

@pytest.fixture
def sample_state():
    """Create sample state for testing"""
    return {
        "pdf_path": "sample_data_paper.pdf",
        "extracted_text": "",
        "text_by_page": {},
        "structure_images": [],
        "plot_images": [],
        "identified_materials": [],
        "battery_stacks": [],
        "processing_params": {},
        "paper_metadata": {},
        "validation_passed": False,
        "detected_structures": [],
        "material_smiles": {},
        "cycling_data": [],
        "voltage_data": [],
        "experiments": [],
        "output_json": {},
        "messages": [],
        "current_agent": "ocr_agent",
        "errors": []
    }

def test_ocr_agent(sample_state):
    """Test OCR agent"""
    if not os.path.exists(sample_state['pdf_path']):
        pytest.skip("Sample PDF not found")
    
    result = ocr_agent(sample_state)
    
    assert result['extracted_text'] != ""
    assert len(result['text_by_page']) > 0
    assert result['current_agent'] == "text_analysis"

def test_text_analysis_agent(sample_state):
    """Test text analysis agent"""
    # Mock extracted text
    sample_state['extracted_text'] = """
    An all-organic aqueous potassium dual-ion battery
    The APDIB was assembled with PTPAn as cathode, PTCDI as anode,
    and 21 M KFSI water-in-salt as the electrolyte.
    """
    
    with patch('ollama.generate') as mock_ollama:
        # Mock responses
        mock_ollama.side_effect = [
            {'response': '{"title": "Test Paper", "authors": ["Author 1"]}'},
            {'response': '[{"full_name": "PTPAn", "role": "cathode"}]'},
            {'response': '{"cathode": {"active_wt": 0.8}}'},
            {'response': '{"current_rate_mAg": 2000}'},
            {'response': '[{"cathode": "PTPAn", "anode": "PTCDI", "electrolyte": ["KFSI"]}]'}
        ]
        
        result = text_analysis_agent(sample_state)
        
        assert result['validation_passed'] == True
        assert len(result['identified_materials']) > 0

def test_state_schema():
    """Test state schema structure"""
    state = AgentState(
        pdf_path="test.pdf",
        extracted_text="",
        text_by_page={},
        structure_images=[],
        plot_images=[],
        identified_materials=[],
        battery_stacks=[],
        processing_params={},
        paper_metadata={},
        validation_passed=False,
        detected_structures=[],
        material_smiles={},
        cycling_data=[],
        voltage_data=[],
        experiments=[],
        output_json={},
        messages=[],
        current_agent="ocr_agent",
        errors=[]
    )
    
    assert state['pdf_path'] == "test.pdf"
    assert isinstance(state['messages'], list)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
