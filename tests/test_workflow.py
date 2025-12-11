"""
Integration tests for complete workflow
"""

import pytest
import os
import json
from workflow.langgraph_workflow import build_workflow

def test_workflow_build():
    """Test workflow can be built"""
    workflow = build_workflow()
    assert workflow is not None

def test_complete_workflow():
    """Test complete workflow execution"""
    pdf_path = "sample_data_paper.pdf"
    
    if not os.path.exists(pdf_path):
        pytest.skip("Sample PDF not found")
    
    from workflow.langgraph_workflow import run_workflow
    
    result = run_workflow(pdf_path)
    
    # Check output
    assert result is not None
    assert 'output_json' in result
    assert 'experiments' in result['output_json']
    
    # Check JSON file was created
    output_file = "outputs/sample_data_paper_extracted.json"
    assert os.path.exists(output_file)
    
    with open(output_file, 'r') as f:
        output = json.load(f)
    
    assert 'paper_info' in output
    assert 'experiments' in output
    assert len(output['experiments']) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
