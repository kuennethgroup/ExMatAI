"""
LangGraph Workflow - ExMat AI Pipeline

DAG:
  ocr_node -> text_extract_node -> ┬-> plots_extract_node  ----------┬-> assembly_node -> END
                                  └-> structure_smiles_node -> smiles_mapping_node -┘
"""

import os
from pathlib import Path

from langgraph.graph import END, StateGraph

from agents.experiment_assembly_agent import assemble_final_json
from agents.ocr_agent import run_deepseek_ocr
from agents.plots_analysis_agent import process_plots
from agents.smiles_mapping_agent import map_smiles_to_materials
from agents.structure_extraction_agent import process_structures
from agents.text_analysis_agent import extract_text_data
from utils.state_schema import WorkflowState


def build_workflow() -> StateGraph:
    """Build and compile the LangGraph workflow."""

    workflow = StateGraph(WorkflowState)

    # -- Add Nodes -------------------------------------------------------
    workflow.add_node("ocr_node", run_deepseek_ocr)
    workflow.add_node("text_extract_node", extract_text_data)
    workflow.add_node("plots_extract_node", process_plots)
    workflow.add_node("structure_smiles_node", process_structures)
    workflow.add_node("smiles_mapping_node", map_smiles_to_materials)
    workflow.add_node("assembly_node", assemble_final_json)

    # -- Define Edges ----------------------------------------------------
    # Sequential: OCR -> Text Extraction
    workflow.set_entry_point("ocr_node")
    workflow.add_edge("ocr_node", "text_extract_node")

    # Fan-out: Text Extraction -> (Plots ‖ Structures+SMILES)
    workflow.add_edge("text_extract_node", "plots_extract_node")
    workflow.add_edge("text_extract_node", "structure_smiles_node")

    # Structure detection -> SMILES mapping
    workflow.add_edge("structure_smiles_node", "smiles_mapping_node")

    # Fan-in: Assembly node only runs after BOTH plots and smiles are done
    workflow.add_edge(["plots_extract_node", "smiles_mapping_node"], "assembly_node")

    # Assembly -> END
    workflow.add_edge("assembly_node", END)

    return workflow.compile()

def run_workflow(pdf_path: str, config_path: str = None) -> dict:
    """
    Run the complete ExMat AI extraction pipeline.

    Args:
        pdf_path: Path to the input PDF file.
        config_path: Optional path to config.yaml (unused for now).

    Returns:
        Final state dict containing the extraction results.
    """
    pdf_name = Path(pdf_path).stem
    output_dir = os.path.join("outputs", pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    initial_state: WorkflowState = {
        "pdf_path": os.path.abspath(pdf_path),
        "output_dir": os.path.abspath(output_dir),
        # Placeholders - populated by nodes
        "mmd_path": None,
        "images_dir": None,
        "experiments_data": [],
        "figures_data": [],
        "extracted_plot_data": {},
        "structure_detections": [],
        "raw_smiles": [],
        "mapped_smiles": {},
        "final_json": {},
        "output_file": "",
    }

    print("\n" + "=" * 70)
    print("  ExMat AI: Starting LangGraph Pipeline")
    print("=" * 70)
    print(f"  PDF:    {pdf_path}")
    print(f"  Output: {output_dir}")
    print("=" * 70 + "\n")

    graph = build_workflow()
    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("  Pipeline execution completed successfully.")
    print(f"  Results saved to: {final_state.get('output_file', 'N/A')}")
    print("=" * 70 + "\n")

    return final_state