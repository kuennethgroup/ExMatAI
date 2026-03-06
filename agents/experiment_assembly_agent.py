"""
Experiment Assembly Agent - Merge all extracted data into final JSON
Last node in the LangGraph pipeline.

Merges:
  - experiments_data  (from Text LLM)
  - extracted_plot_data  (CSV paths + metadata from plot extraction)
  - mapped_smiles  (SMILES mapped to material names)

Writes the final output JSON.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from utils.state_schema import WorkflowState


def assemble_final_json(state: WorkflowState) -> WorkflowState:
    """LangGraph node: merge all data and produce the final JSON output."""
    print("\n" + "=" * 70)
    print("Step 6: Assembling Final JSON Output")
    print("=" * 70)

    experiments_data = state.get("experiments_data", [])
    extracted_plot_data = state.get("extracted_plot_data", {})
    mapped_smiles = state.get("mapped_smiles", {})
    output_dir = state["output_dir"]

    # -- 1. Enrich each experiment with SMILES + Plot Data --------------
    for exp in experiments_data:
        # Attach SMILES for negative electrode
        neg_name = exp.get("Material_Name_Negative")
        if neg_name and neg_name in mapped_smiles:
            exp["SMILES_Negative"] = mapped_smiles[neg_name]
        else:
            exp["SMILES_Negative"] = None

        # Attach SMILES for positive electrode
        pos_name = exp.get("Material_Name_Positive")
        if pos_name and pos_name in mapped_smiles:
            exp["SMILES_Positive"] = mapped_smiles[pos_name]
        else:
            exp["SMILES_Positive"] = None

        # Attach cycle data
        cycle_figs = exp.get("Cycle_Data_Figure", [])
        if not isinstance(cycle_figs, list):
            cycle_figs = [cycle_figs] if cycle_figs else []
        
        cycle_data_list = []
        for cycle_fig in cycle_figs:
            if not cycle_fig: continue
            
            # Normalize cycle fig (e.g. "Fig. 3b" -> "3b")
            norm_cycle = cycle_fig
            import re
            m = re.match(r'(?:Fig(?:ure|\.)?\s*)(\d+)\s*([a-zA-Z])?', cycle_fig, re.IGNORECASE)
            if m:
                fig_num = m.group(1)
                sub_label = (m.group(2) or "").lower()
                norm_cycle = f"{fig_num}{sub_label}"
            
            # Try exact match, then parent figure
            plot_entry = extracted_plot_data.get(norm_cycle)
            if not plot_entry and fig_num:
                plot_entry = extracted_plot_data.get(fig_num)
                
            if plot_entry:
                cycle_data_list.append({
                    "figure": cycle_fig,
                    "csv_path": plot_entry.get("csv_path"),
                    "axis_metadata": plot_entry.get("metadata", {}),
                })
        exp["Cycle_Data"] = cycle_data_list if cycle_data_list else None

        # Attach voltage profile data
        voltage_figs = exp.get("Voltage_Profile_Figure", [])
        if not isinstance(voltage_figs, list):
            voltage_figs = [voltage_figs] if voltage_figs else []
            
        voltage_data_list = []
        for voltage_fig in voltage_figs:
            if not voltage_fig: continue
            
            # Normalize voltage fig (e.g. "Fig. 3b" -> "3b")
            norm_voltage = voltage_fig
            import re
            m = re.match(r'(?:Fig(?:ure|\.)?\s*)(\d+)\s*([a-zA-Z])?', voltage_fig, re.IGNORECASE)
            if m:
                fig_num = m.group(1)
                sub_label = (m.group(2) or "").lower()
                norm_voltage = f"{fig_num}{sub_label}"
                
            plot_entry = extracted_plot_data.get(norm_voltage)
            if not plot_entry and fig_num:
                plot_entry = extracted_plot_data.get(fig_num)
                
            if plot_entry:
                voltage_data_list.append({
                    "figure": voltage_fig,
                    "csv_path": plot_entry.get("csv_path"),
                    "axis_metadata": plot_entry.get("metadata", {}),
                })
        exp["Voltage_Profile_Data"] = voltage_data_list if voltage_data_list else None

    # -- 2. Build final output ------------------------------------------
    pdf_name = Path(state["pdf_path"]).stem

    final_json = {
        "paper_name": pdf_name,
        "extraction_metadata": {
            "extraction_date": datetime.now().isoformat(),
            "pdf_path": state["pdf_path"],
            "models_used": {
                "ocr": "DeepSeek-OCR (vLLM)",
                "text_llm": "qwen3.5:35b",
                "vision_llm": "qwen3-vl:32b",
                "structure_detection": "UniParser/MolDetv2",
                "smiles_generation": "MolNexTR",
            },
            "total_experiments": len(experiments_data),
            "total_smiles_mapped": len(mapped_smiles),
            "total_plots_extracted": len(extracted_plot_data),
        },
        "experiments": experiments_data,
    }

    # -- 3. Write output ------------------------------------------------
    output_file = os.path.join(output_dir, f"{pdf_name}_extracted.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

    print(f"  Total Experiments Found:    {len(experiments_data)}")
    print(f"  Total SMILES Mapped:        {len(mapped_smiles)}")
    print(f"  Total Plot Data Extracted:  {len(extracted_plot_data)}")
    print(f"  Final Output Saved To:      {output_file}")

    return {"final_json": final_json, "output_file": output_file}