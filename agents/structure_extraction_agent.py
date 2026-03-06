"""
Structure Extraction Agent - Detect Structures + Generate SMILES
Absorbs logic from modules/structures_det.py and modules/smiles_gen.py.

Pipeline:
  1. Filter images that contain chemical structure references (from experiments_data)
  2. Run MolDetv2 (YOLO) to detect structure bounding boxes
  3. Crop each detected region
  4. Run MolNexTR on each crop to generate SMILES strings
  5. Return structure_detections and raw_smiles to state
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

from utils.state_schema import WorkflowState

# =======================================================================
# Helpers
# =======================================================================

_model_cache: Optional[YOLO] = None


def _get_yolo_model() -> YOLO:
    """Download & cache the MolDetv2 YOLO model."""
    global _model_cache
    if _model_cache is None:
        model_path = hf_hub_download(
            repo_id="UniParser/MolDetv2",
            filename="moldet_v2_yolo11n_640_general.pt",
            repo_type="model",
        )
        _model_cache = YOLO(model_path)
    return _model_cache


def _extract_fig_number(ref: str) -> Optional[str]:
    """Extract the numeric part from any figure reference variant."""
    m = re.search(r'(?:Fig(?:ure|\.)?)\s*(\d+)', ref, re.IGNORECASE)
    return m.group(1) if m else None


def _normalize_fig_ref(ref: Optional[str]):
    """Normalize any figure reference to (number, sub_label)."""
    if not ref:
        return None
    m = re.match(r'(?:Fig(?:ure|\.)?)\s*(\d+)\s*([a-zA-Z])?', ref.strip(), re.IGNORECASE)
    if m:
        return m.group(1), (m.group(2) or "").lower()
    return None


def _find_image_for_figure(fig_number: str, figures_data: list, images_dir: str) -> Optional[str]:
    """Given a figure number (e.g. '3'), find the matching image path."""
    for fig in figures_data:
        fig_num = _extract_fig_number(fig["Figure_ID"])
        if fig_num == fig_number:
            rel_path = fig["Image_Path"]
            img_name = os.path.basename(rel_path)
            full_path = os.path.join(images_dir, img_name)
            if os.path.exists(full_path):
                return full_path
            alt = os.path.join(os.path.dirname(images_dir), rel_path.lstrip("./"))
            if os.path.exists(alt):
                return alt
    return None


# =======================================================================
# LangGraph Node
# =======================================================================

def process_structures(state: WorkflowState) -> WorkflowState:
    """LangGraph node: detect chemical structures and generate SMILES."""
    print("\n" + "=" * 70)
    print("Step 4: Detecting chemical structures and generating SMILES")
    print("=" * 70)

    experiments_data = state["experiments_data"]
    figures_data = state["figures_data"]
    images_dir = state["images_dir"]
    output_dir = state["output_dir"]

    structures_dir = os.path.join(output_dir, "structures")
    os.makedirs(structures_dir, exist_ok=True)

    # -- 1. Collect structure figure references from experiments ---------
    target_refs: Dict[str, List[str]] = {}
    for exp in experiments_data:
        for key in ("Structure_Figure_Negative", "Structure_Figure_Positive"):
            ref = exp.get(key)
            parsed = _normalize_fig_ref(ref)
            if parsed:
                fig_number, sub_label = parsed
                target_refs.setdefault(fig_number, [])
                if sub_label and sub_label not in target_refs[fig_number]:
                    target_refs[fig_number].append(sub_label)

    print(f"  Identified structure figures to process: {list(target_refs.keys())}")

    if not target_refs:
        print("  No chemical structures explicitly identified in metadata text.")
        print("  FALLBACK: Scanning all document images with MolDetV2 for orphaned structures...")
        for fig in figures_data:
             fig_num = _extract_fig_number(fig["Figure_ID"])
             if fig_num:
                 target_refs.setdefault(fig_num, [])

    # -- 2. Load YOLO model ---------------------------------------------
    print("  Loading MolDetv2 object detection model...")
    model = _get_yolo_model()

    structure_detections: List[Dict] = []
    raw_smiles: List[Dict] = []

    # -- 3. Process each target figure ----------------------------------
    for fig_id, sub_labels in target_refs.items():
        img_path = _find_image_for_figure(fig_id, figures_data, images_dir)
        if not img_path:
            print(f"  Could not find image file for {fig_id}")
            continue

        print(f"  Processing structure annotations in {fig_id} ({img_path})...")

        # If sub-labels, try figpanel first
        images_to_process = []
        if sub_labels:
            try:
                import figpanel
                panels = figpanel.extract(img_path, output_dir=None)
                for panel in panels:
                    panel_label = (panel.label or "").lower()
                    if panel_label in sub_labels:
                        crop_path = os.path.join(structures_dir, f"{fig_id.replace(' ', '_')}_{panel_label}_panel.png")
                        panel.image.save(crop_path)
                        images_to_process.append((f"{fig_id}{panel_label}", crop_path))
            except Exception as e:
                print(f"  figpanel cropping failed ({e}), defaulting to full image processing.")
                images_to_process.append((fig_id, img_path))
        else:
            images_to_process.append((fig_id, img_path))

        for ref_tag, proc_img_path in images_to_process:
            # -- YOLO Detection -----------------------------------------
            results = model.predict(
                proc_img_path,
                save=True,
                project=structures_dir,
                name=ref_tag.replace(" ", "_").replace(".", ""),
                imgsz=640,
                conf=0.5,
                device=0,
            )

            boxes = []
            smiles_list = []
            original_img = cv2.imread(proc_img_path)

            for r in results:
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

                    # -- Crop & generate SMILES -------------------------
                    crop = original_img[y1:y2, x1:x2]
                    crop_path = os.path.join(structures_dir, f"{ref_tag.replace(' ', '_')}_struct_{i}.png")
                    cv2.imwrite(crop_path, crop)

                    try:
                        import MolNexTR
                        from rdkit import Chem

                        preds = MolNexTR.get_predictions(crop_path, smiles=True)
                        smiles = preds.get("predicted_smiles", "")

                        # Validate SMILES with RDKit
                        if smiles and isinstance(smiles, str) and smiles.strip():
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None:
                                smiles_list.append({"crop_path": crop_path, "smiles": smiles, "box": boxes[-1]})
                                print(f"  Valid SMILES for {ref_tag} (structure {i}): {smiles[:60]}..." if len(smiles) > 60 else f"  Valid SMILES for {ref_tag} (structure {i}): {smiles}")
                            else:
                                print(f"  Invalid SMILES generated for {ref_tag} (structure {i}), discarding.")
                                smiles_list.append({"crop_path": crop_path, "smiles": "", "box": boxes[-1]})
                        else:
                            print(f"  MolNexTR failed to generate SMILES for {ref_tag} (structure {i}).")
                            smiles_list.append({"crop_path": crop_path, "smiles": "", "box": boxes[-1]})

                    except Exception as e:
                        print(f"  SMILES generation failed for structure {i}: {e}")
                        smiles_list.append({"crop_path": crop_path, "smiles": "", "box": boxes[-1]})

            # Save annotated image path (YOLO saves automatically)
            annotated_dir = os.path.join(structures_dir, ref_tag.replace(" ", "_").replace(".", ""))
            annotated_path = ""
            if os.path.isdir(annotated_dir):
                annotated_files = list(Path(annotated_dir).glob("*.jpg")) + list(Path(annotated_dir).glob("*.png"))
                if annotated_files:
                    annotated_path = str(annotated_files[0])

            structure_detections.append({
                "ref_tag": ref_tag,
                "original_image": img_path,
                "processed_image": proc_img_path,
                "annotated_image": annotated_path,
                "boxes": boxes,
            })

            raw_smiles.append({
                "ref_tag": ref_tag,
                "image_path": proc_img_path,
                "smiles_list": smiles_list,
            })

    total_smiles = sum(len(s['smiles_list']) for s in raw_smiles)
    valid_smiles = sum(1 for s in raw_smiles for entry in s['smiles_list'] if entry.get('smiles'))
    print(f"  Finished processing: Detected {len(structure_detections)} total chemical structures and successfully validated {valid_smiles}/{total_smiles} SMILES strings.")

    return {
        "structure_detections": structure_detections,
        "raw_smiles": raw_smiles,
    }