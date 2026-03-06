"""
ExMat AI - Workflow State Schema
Defines the TypedDict used by all LangGraph nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional


class WorkflowState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────────────────
    pdf_path: str
    output_dir: str

    # ── OCR Outputs ─────────────────────────────────────────────────────
    mmd_path: Optional[str]
    images_dir: Optional[str]

    # ── Text Extraction Outputs ─────────────────────────────────────────
    experiments_data: List[Dict[str, Any]]       # Structured experiments from Text LLM
    figures_data: List[Dict[str, Any]]           # [{Figure_ID, Image_Path, Caption}, ...]

    # ── Plot Extraction Outputs ─────────────────────────────────────────
    extracted_plot_data: Dict[str, Any]           # {fig_ref: {csv_path, metadata, ...}}

    # ── Structure + SMILES Outputs ──────────────────────────────────────
    structure_detections: List[Dict[str, Any]]    # [{image_path, boxes, annotated_path}, ...]
    raw_smiles: List[Dict[str, Any]]              # [{image_path, smiles_list}, ...]

    # ── SMILES Mapping Outputs ──────────────────────────────────────────
    mapped_smiles: Dict[str, Any]                 # {material_name: smiles_string}

    # ── Final Output ────────────────────────────────────────────────────
    final_json: Dict[str, Any]
    output_file: str