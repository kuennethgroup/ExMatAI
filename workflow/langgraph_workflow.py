"""
LangGraph Workflow - Complete 7-step pipeline
"""

from langgraph.graph import StateGraph, END
from utils.state_schema import AgentState

# Import all agents
from agents.ocr_agent import ocr_agent
from agents.text_analysis_agent import text_analysis_agent
from agents.structure_extraction_agent import structure_extraction_agent
from agents.smiles_generation_agent import smiles_generation_agent
from agents.plots_analysis_agent import plots_analysis_agent
from agents.experiment_assembly_agent import experiment_assembly_agent
from agents.data_aggregation_agent import data_aggregation_agent

def build_workflow():
    """
    Build the complete LangGraph workflow with all 7 agents
    
    Workflow:
    1. OCR (DeepSeek-OCR)
    2. Text Analysis (Qwen3:8b)
    3. Structure Detection (MolDetV2) + Labeling (Qwen3-VL)
    4. SMILES Generation (ChemVLM)
    5. Plot Analysis (Qwen3-VL)
    6. Experiment Assembly
    7. Data Aggregation & Export
    """
    
    # Create workflow graph
    workflow = StateGraph(AgentState)
    
    # Add all agent nodes
    workflow.add_node("ocr_agent", ocr_agent)
    workflow.add_node("text_analysis", text_analysis_agent)
    workflow.add_node("structure_extraction", structure_extraction_agent)
    workflow.add_node("smiles_generation", smiles_generation_agent)
    workflow.add_node("plots_analysis", plots_analysis_agent)
    workflow.add_node("experiment_assembly", experiment_assembly_agent)
    workflow.add_node("data_aggregation", data_aggregation_agent)
    
    # Define workflow edges (sequential pipeline)
    workflow.set_entry_point("ocr_agent")
    
    workflow.add_edge("ocr_agent", "text_analysis")
    workflow.add_edge("text_analysis", "structure_extraction")
    workflow.add_edge("structure_extraction", "smiles_generation")
    workflow.add_edge("smiles_generation", "plots_analysis")
    workflow.add_edge("plots_analysis", "experiment_assembly")
    workflow.add_edge("experiment_assembly", "data_aggregation")
    workflow.add_edge("data_aggregation", END)
    
    return workflow.compile()

def run_workflow(pdf_path: str, config_path: str = None):
    """
    Run the complete extraction workflow
    
    Args:
        pdf_path: Path to input PDF
        config_path: Optional config file path
        
    Returns:
        Final state with all extracted data
    """
    
    # Build workflow
    app = build_workflow()
    
    # Initial state
    initial_state = {
        "pdf_path": pdf_path,
        "config_path": config_path,
        "extracted_text": "",
        "text_by_page": {},
        "structure_images": [],
        "plot_images": [],
        "paper_info": {},
        "identified_materials": [],
        "detected_structures": [],
        "material_smiles": {},
        "cycling_data": [],
        "voltage_data": [],
        "battery_stacks": [],
        "processing_params": {},
        "experiments": [],
        "errors": [],
        "messages": [],
        "current_agent": "ocr_agent"
    }
    
    # Run workflow
    result = app.invoke(initial_state)
    
    return result
