"""
SMILES Mapping Agent - Vision LLM maps SMILES to Material Names
Uses qwen3-vl:32b to look at the full structure image with annotated bounding
boxes, read the structure names, and map each SMILES to its material name.

Input from state:
  - structure_detections  (annotated images with bounding boxes)
  - raw_smiles            (SMILES strings per crop)
  - figures_data          (captions for context)
  - experiments_data      (material names for cross-referencing)

Output to state:
  - mapped_smiles  {material_name: smiles_string}
"""

import base64
import json
import os
from typing import Dict, List

import requests

from utils.state_schema import WorkflowState

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
VISION_MODEL = "qwen3-vl:32b"


def _encode_image(path: str) -> str:
    """Read an image file and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_caption_for_figure(ref_tag: str, figures_data: list) -> str:
    """Find the caption matching a ref_tag like 'Fig. 1' or 'Fig. 1a'."""
    # Strip sub-label for parent figure match
    for fig in figures_data:
        if fig["Figure_ID"] in ref_tag or ref_tag.startswith(fig["Figure_ID"].replace(" ", "")):
            return fig.get("Caption", "")
    return ""


def map_smiles_to_materials(state: WorkflowState) -> WorkflowState:
    """LangGraph node: use Vision LLM to map SMILES to material names."""
    print("\n" + "=" * 70)
    print("Step 5: Mapping SMILES to material names using Vision LLM")
    print("=" * 70)

    structure_detections = state.get("structure_detections", [])
    raw_smiles = state.get("raw_smiles", [])
    figures_data = state.get("figures_data", [])
    experiments_data = state.get("experiments_data", [])

    if not raw_smiles:
        print("  No SMILES data available to map. Skipping this step.")
        return {"mapped_smiles": {}}

    # Collect all known material names from experiments
    known_materials: List[str] = []
    for exp in experiments_data:
        for key in ("Material_Name_Negative", "Material_Name_Positive"):
            name = exp.get(key)
            if name and name not in known_materials:
                known_materials.append(name)

    print(f"  Known materials from the text extraction: {known_materials}")

    mapped_smiles: Dict[str, str] = {}

    for smiles_entry in raw_smiles:
        ref_tag = smiles_entry["ref_tag"]
        image_path = smiles_entry["image_path"]
        smiles_list = smiles_entry["smiles_list"]

        if not smiles_list:
            continue

        # Find corresponding detection for annotated image
        det = next((d for d in structure_detections if d["ref_tag"] == ref_tag), None)
        caption = _get_caption_for_figure(ref_tag, figures_data)

        # Build the SMILES summary
        smiles_summary = []
        for i, s in enumerate(smiles_list):
            smiles_summary.append(f"  Structure {i+1}: SMILES = {s['smiles']}")
        smiles_text = "\n".join(smiles_summary)

        # Build images list for vision LLM
        images_b64 = [_encode_image(image_path)]
        if det and det.get("annotated_image") and os.path.exists(det["annotated_image"]):
            images_b64.append(_encode_image(det["annotated_image"]))

        prompt = f"""You are a chemistry expert. I have a scientific figure containing chemical structures.

Figure caption: {caption}

The following chemical structures were detected in this image with their SMILES:
{smiles_text}

Known material names from this paper: {', '.join(known_materials)}

Tasks:
1. Look at the image and its annotated bounding boxes
2. Identify the name/label of each structure visible in the image
3. Map each SMILES string to its material name

Output strictly valid JSON (no markdown):
{{
  "mappings": [
    {{"material_name": "...", "smiles": "...", "confidence": 0.95}},
    ...
  ]
}}
"""
        print(f"  Querying Vision LLM to map {len(smiles_list)} structures from {ref_tag}...")

        try:
            resp = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": VISION_MODEL,
                    "messages": [{"role": "user", "content": prompt, "images": images_b64}],
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=300,
            )
            raw_text = resp.json()["message"]["content"].strip()

            # Parse JSON
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            result = json.loads(raw_text)
            for mapping in result.get("mappings", []):
                mat_name = mapping.get("material_name", "")
                smiles = mapping.get("smiles", "")
                if mat_name and smiles:
                    mapped_smiles[mat_name] = smiles
                    print(f"  Successfully mapped {mat_name} -> {smiles[:50]}..." if len(smiles) > 50 else f"  Successfully mapped {mat_name} -> {smiles}")

        except Exception as e:
            print(f"  Vision LLM mapping failed for {ref_tag}: {e}")
            # Fallback: assign SMILES without naming
            for s in smiles_list:
                if s["smiles"]:
                    mapped_smiles[f"unknown_structure_{ref_tag}"] = s["smiles"]

    print(f"  Finished: Mapped a total of {len(mapped_smiles)} SMILES strings to specific materials.")

    return {"mapped_smiles": mapped_smiles}
