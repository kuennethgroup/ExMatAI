"""
Plots Analysis Agent - Filter, Crop & Extract Plot Data
Absorbs logic from modules/plots_det.py and modules/plots_extract.py.

Pipeline:
  1. Filter relevant plot images using figure references from experiments_data
  2. Use figpanel to crop sub-panels from composite figures
  3. Run FastChartExtractor (OpenCV + Vision LLM) on each relevant sub-panel
  4. Save CSV data and return extracted_plot_data to state
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from utils.state_schema import WorkflowState

# =======================================================================
# FastChartExtractor (from modules/plots_extract.py)
# =======================================================================


class FastChartExtractor:
    """Extract numerical data from chart images using OpenCV + Vision LLM."""

    def __init__(self, ollama_url: str = "http://localhost:11434/api/chat",
                 ollama_model: str = "qwen3-vl:32b"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    # ------ colour masks ----------------------------------------------
    @staticmethod
    def _get_color_mask(hsv_img, color_name: str):
        c = color_name.lower()
        if "red" in c:
            m1 = cv2.inRange(hsv_img, np.array([0, 40, 40]), np.array([12, 255, 255]))
            m2 = cv2.inRange(hsv_img, np.array([160, 40, 40]), np.array([180, 255, 255]))
            return m1 | m2
        if "blue" in c or "cyan" in c or "teal" in c:
            return cv2.inRange(hsv_img, np.array([80, 40, 40]), np.array([130, 255, 255]))
        if "green" in c:
            return cv2.inRange(hsv_img, np.array([40, 40, 40]), np.array([85, 255, 255]))
        if "orange" in c or "yellow" in c:
            return cv2.inRange(hsv_img, np.array([10, 50, 50]), np.array([35, 255, 255]))
        if "black" in c or "grey" in c or "gray" in c:
            return cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([180, 50, 100]))
        return cv2.inRange(hsv_img, np.array([0, 40, 40]), np.array([180, 255, 255]))

    # ------ main extraction -------------------------------------------
    def run(self, image_path: str, output_dir: str, tag: str = "") -> Dict:
        """Run full extraction pipeline on a single chart image.

        Returns dict with keys: csv_path, plot_path, metadata.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(inv, 50, 255, cv2.THRESH_BINARY)

        h, w = img.shape[:2]

        # Morphological ops to find the plot area bounding box
        horz = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.4), 1)),
        )
        vert = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.4))),
        )

        contours, _ = cv2.findContours(horz + vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plot_box = None
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            if cw > w * 0.3 and ch > h * 0.3:
                plot_box = [cx, cy, cx + cw, cy + ch]
                break
        if not plot_box:
            plot_box = [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)]

        # -- Vision LLM: axis metadata ----------------------------------
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        vlm_prompt = """
Analyze this chart. Output strictly valid JSON with no markdown formatting.
We need the min/max numerical values on the axes, the titles of each axis, and the color/axis assignments for each line.
Format EXACTLY like this example:
{
  "ranges": {
    "x": {"min": 0, "max": 1000, "title": "Cycle number (N)"},
    "left_y": {"min": 0, "max": 750, "title": "Specific capacity (mAh g-1)"},
    "right_y": {"min": 0, "max": 100, "title": "Coulombic efficiency (%)"}
  },
  "series": [
    {"color": "red", "axis": "left", "label": "1st cycle"},
    {"color": "teal", "axis": "left", "label": "2nd cycle"}
  ]
}
"""
        resp = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "messages": [{"role": "user", "content": vlm_prompt, "images": [img_b64]}],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=300,
        )
        raw_text = resp.json()["message"]["content"].strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
        metadata = json.loads(raw_text)

        # -- Pixel-to-data extraction -----------------------------------
        x_min, y_min, x_max, y_max = plot_box
        plot_crop = img[y_min:y_max, x_min:x_max]
        hsv_crop = cv2.cvtColor(plot_crop, cv2.COLOR_BGR2HSV)
        px_w, px_h = x_max - x_min, y_max - y_min

        r_x = metadata["ranges"].get("x", {"min": 0, "max": 100, "title": "X-Axis"})
        x_col_title = r_x.get("title", "X-Axis")

        df_combined = pd.DataFrame({"x_px": np.arange(px_w)})
        df_combined[x_col_title] = r_x["min"] + (df_combined["x_px"] / px_w) * (r_x["max"] - r_x["min"])

        fig_plot, ax_left = plt.subplots(figsize=(10, 5))
        has_right = any("right" in s.get("axis", "").lower() for s in metadata.get("series", []))
        ax_right = ax_left.twinx() if has_right else None
        lines, labels = [], []

        for series in metadata.get("series", []):
            color = series.get("color", "unknown")
            label = series.get("label", "unlabeled")
            axis = series.get("axis", "left").lower()

            mask = self._get_color_mask(hsv_crop, color)
            points_upper, points_lower = [], []

            for xv in range(px_w):
                y_indices = np.where(mask[:, xv])[0]
                if len(y_indices) == 0:
                    continue
                splits = np.where(np.diff(y_indices) > 15)[0] + 1
                clusters = np.split(y_indices, splits)
                points_upper.append({"x_px": xv, "y_px": clusters[0].mean()})
                if len(clusters) > 1:
                    points_lower.append({"x_px": xv, "y_px": clusters[-1].mean()})

            if not points_upper:
                continue

            y_axis_key = f"{axis}_y"
            r_y = metadata["ranges"].get(y_axis_key, metadata["ranges"].get("left_y", {"min": 0, "max": 100, "title": "Y-Axis"}))
            y_col_title = r_y.get("title", f"{axis.capitalize()} Y-Axis")
            base_col = y_col_title if (label.lower() in y_col_title.lower() or label == "unlabeled") else f"{y_col_title} ({label})"

            df_up = pd.DataFrame(points_upper)
            df_up["x_data"] = r_x["min"] + (df_up["x_px"] / px_w) * (r_x["max"] - r_x["min"])
            df_up["y_data"] = r_y["min"] + ((px_h - df_up["y_px"]) / px_h) * (r_y["max"] - r_y["min"])

            target_ax = ax_right if ("right" in axis and ax_right) else ax_left
            pc = "darkorange" if color == "orange" else ("darkcyan" if "teal" in color or "cyan" in color else color)
            try:
                line, = target_ax.plot(df_up["x_data"], df_up["y_data"], color=pc, linewidth=2, label=label)
            except ValueError:
                line, = target_ax.plot(df_up["x_data"], df_up["y_data"], color="black", linewidth=2, label=label)
            lines.append(line)
            labels.append(label)

            if len(points_lower) > len(points_upper) * 0.1:
                df_low = pd.DataFrame(points_lower)
                df_low["x_data"] = r_x["min"] + (df_low["x_px"] / px_w) * (r_x["max"] - r_x["min"])
                df_low["y_data"] = r_y["min"] + ((px_h - df_low["y_px"]) / px_h) * (r_y["max"] - r_y["min"])
                target_ax.plot(df_low["x_data"], df_low["y_data"], color=pc, linewidth=2, label="_nolegend_")
                up_m = df_up[["x_px", "y_data"]].rename(columns={"y_data": f"{base_col} (Top Curve)"})
                lo_m = df_low[["x_px", "y_data"]].rename(columns={"y_data": f"{base_col} (Bottom Curve)"})
                df_combined = pd.merge(df_combined, up_m, on="x_px", how="left")
                df_combined = pd.merge(df_combined, lo_m, on="x_px", how="left")
            else:
                up_m = df_up[["x_px", "y_data"]].rename(columns={"y_data": base_col})
                df_combined = pd.merge(df_combined, up_m, on="x_px", how="left")

        # Finalize CSV
        df_combined = df_combined.drop(columns=["x_px"])
        for col in df_combined.columns:
            df_combined[col] = df_combined[col].round(2)

        safe_tag = re.sub(r'[^\w\-]', '_', tag) if tag else Path(image_path).stem
        csv_path = os.path.join(output_dir, f"{safe_tag}_data.csv")
        df_combined.to_csv(csv_path, index=False)

        # Finalize reconstructed plot
        ax_left.set_xlabel(x_col_title)
        left_y_info = metadata["ranges"].get("left_y", {})
        ax_left.set_ylabel(left_y_info.get("title", "Left Y-Axis"))
        if "min" in r_x and "max" in r_x:
            ax_left.set_xlim(r_x["min"], r_x["max"])
        if "min" in left_y_info and "max" in left_y_info:
            ax_left.set_ylim(left_y_info["min"], left_y_info["max"])
        if ax_right:
            right_y_info = metadata["ranges"].get("right_y", {})
            ax_right.set_ylabel(right_y_info.get("title", "Right Y-Axis"))
            if "min" in right_y_info and "max" in right_y_info:
                ax_right.set_ylim(right_y_info["min"], right_y_info["max"])
        if lines:
            ax_left.legend(lines, labels, loc="best")
        plt.title("Reconstructed Chart")
        ax_left.grid(True, linestyle="--", alpha=0.5)
        plot_path = os.path.join(output_dir, f"{safe_tag}_reconstructed.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close(fig_plot)

        return {"csv_path": csv_path, "plot_path": plot_path, "metadata": metadata}


# =======================================================================
# Helpers
# =======================================================================

_EXPLICIT_PLOT_PHRASES = [
    "voltage profile", "charge-discharge", "charge/discharge", "galvanostatic",
    "rate performance", "rate capability", "cycling performance",
    "cycle stability", "cyclic performance", "capacity retention",
    "cv curves", "cyclic voltammetry"
]

_Y_KEYWORDS = ["capacity", "coulombic", "energy density", "power density"]
_X_KEYWORDS = ["voltage", "cycle", "current density", "c-rate", "rate"]

def _is_relevant_plot_text(text: str) -> bool:
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in _EXPLICIT_PLOT_PHRASES):
        return True
    
    has_y = any(kw in text_lower for kw in _Y_KEYWORDS)
    # Match specific words for X-axis: 
    # re.search ensures that 'v' isn't just part of a word like 'curve'
    has_x = any(kw in text_lower for kw in _X_KEYWORDS) or re.search(r'\bv\b', text_lower)
    
    return has_y and has_x


def _extract_fig_number(ref: str) -> Optional[str]:
    """Extract the numeric part from any figure reference variant.
    'Fig 3a' -> '3', 'Fig. 3b' -> '3', 'Figure 3' -> '3'
    """
    m = re.search(r'(?:Fig(?:ure|\.)?\s*)(\d+)', ref, re.IGNORECASE)
    return m.group(1) if m else None


def _normalize_fig_ref(ref: Optional[str]):
    """Normalize any figure reference to (number, sub_label).
    'Fig 3a' -> ('3', 'a'), 'Fig. 3b' -> ('3', 'b'), 'Figure 3' -> ('3', '')
    """
    if not ref:
        return None
    ref = ref.strip()
    m = re.match(r'(?:Fig(?:ure|\.)?\s*)(\d+)\s*([a-zA-Z])?', ref, re.IGNORECASE)
    if m:
        return m.group(1), (m.group(2) or "").lower()
    return None


def _find_image_for_figure(fig_number: str, figures_data: list, images_dir: str) -> Optional[str]:
    """Given a figure number (e.g. '3'), find the matching image path.
    Matches against Figure_ID like 'Figure 3', 'Fig. 3', etc.
    """
    for fig in figures_data:
        fig_num = _extract_fig_number(fig["Figure_ID"])
        if fig_num == fig_number:
            rel_path = fig["Image_Path"]
            img_name = os.path.basename(rel_path)
            full_path = os.path.join(images_dir, img_name)
            if os.path.exists(full_path):
                return full_path
            # Try resolving relative to output_dir parent
            alt = os.path.join(os.path.dirname(images_dir), rel_path.lstrip("./"))
            if os.path.exists(alt):
                return alt
    return None


# =======================================================================
# LangGraph Node
# =======================================================================

def process_plots(state: WorkflowState) -> WorkflowState:
    """LangGraph node: extract numerical data from relevant plot images."""
    print("\n" + "=" * 70)
    print("Step 3: Extracting plot data using Vision LLM and OpenCV")
    print("=" * 70)

    experiments_data = state["experiments_data"]
    figures_data = state["figures_data"]
    images_dir = state["images_dir"]
    output_dir = state["output_dir"]

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # -- 1. Collect figure references from experiments -------------------
    target_refs: Dict[str, List[str]] = {}  # fig_number -> [sub_labels]
    for exp in experiments_data:
        for key in ("Cycle_Data_Figure", "Voltage_Profile_Figure"):
            refs = exp.get(key, [])
            if not isinstance(refs, list):
                refs = [refs] if refs else []
            for ref in refs:
                if not ref: continue
                parsed = _normalize_fig_ref(ref)
                if parsed:
                    fig_number, sub_label = parsed
                    target_refs.setdefault(fig_number, [])
                    if sub_label and sub_label not in target_refs[fig_number]:
                        target_refs[fig_number].append(sub_label)

    # Also keyword-match captions AND document text context
    full_text = ""
    if state.get("mmd_path") and os.path.exists(state["mmd_path"]):
        with open(state["mmd_path"], "r", encoding="utf-8") as f:
            full_text = f.read()

    for fig in figures_data:
        caption = fig.get("Caption", "")
        fig_num = _extract_fig_number(fig["Figure_ID"])
        if not fig_num: continue
        
        kw_found = False
        if _is_relevant_plot_text(caption):
            kw_found = True
        elif full_text:
            # Check surrounding text in the paper (e.g. 250 chars before and after mention)
            for mention in re.finditer(r'fig(?:ure|\.)?\s*' + str(fig_num) + r'\b', full_text, re.IGNORECASE):
                start = max(0, mention.start() - 250)
                end = min(len(full_text), mention.end() + 250)
                context = full_text[start:end]
                if _is_relevant_plot_text(context):
                    kw_found = True
                    break
                    
        if kw_found and fig_num not in target_refs:
            target_refs[fig_num] = []

    print(f"  Identified plots to extract: {list(target_refs.keys())}")

    # -- 2. Process each target figure ----------------------------------
    extractor = FastChartExtractor()
    extracted_plot_data: Dict[str, dict] = {}

    for fig_number, sub_labels in target_refs.items():
        img_path = _find_image_for_figure(fig_number, figures_data, images_dir)
        if not img_path:
            print(f"  Could not find image file for {fig_number}")
            continue

        print(f"  Processing figure {fig_number} from {img_path}...")

        # If sub-labels requested, try figpanel cropping
        if sub_labels:
            matched_any = False
            try:
                import figpanel
                panels = figpanel.extract(img_path, output_dir=None)
                for panel in panels:
                    panel_label = (panel.label or "").lower()
                    if panel_label in sub_labels:
                        matched_any = True
                        # Save cropped panel
                        crop_path = os.path.join(plots_dir, f"fig{fig_number}_{panel_label}.png")
                        panel.image.save(crop_path)
                        tag = f"fig{fig_number}_{panel_label}"
                        try:
                            result = extractor.run(crop_path, plots_dir, tag=tag)
                            extracted_plot_data[f"{fig_number}{panel_label}"] = result
                            print(f"  Successfully extracted data for panel {fig_number}{panel_label}. Saved to: {result['csv_path']}")
                        except Exception as e:
                            print(f"  Failed to extract data for panel {fig_number}{panel_label}: {e}")

                # Fallback: if figpanel didn't match any requested sub-labels
                if not matched_any:
                    print(f"  Could not find matching sub-panels using figpanel; processing the entire figure for {fig_number}")
                    tag = f"fig{fig_number}_full"
                    try:
                        result = extractor.run(img_path, plots_dir, tag=tag)
                        extracted_plot_data[fig_number] = result
                        print(f"  Successfully extracted data for the full figure {fig_number}. Saved to: {result['csv_path']}")
                    except Exception as e:
                        print(f"  Failed to extract data for the full figure {fig_number}: {e}")

            except ImportError:
                print("  figpanel is not installed; processing the entire figure.")
                tag = f"fig{fig_number}_full"
                try:
                    result = extractor.run(img_path, plots_dir, tag=tag)
                    extracted_plot_data[fig_number] = result
                    print(f"  Successfully extracted data for figure {fig_number}. Saved to: {result['csv_path']}")
                except Exception as e:
                    print(f"  Failed to extract data for figure {fig_number}: {e}")
            except Exception as e:
                print(f"  figpanel cropping failed ({e}), defaulting to full image processing.")
                tag = f"fig{fig_number}_full"
                try:
                    result = extractor.run(img_path, plots_dir, tag=tag)
                    extracted_plot_data[fig_number] = result
                except Exception as e2:
                    print(f"  Extraction failed: {e2}")
        else:
            # No sub-label -> process full image
            tag = f"fig{fig_number}_full"
            try:
                result = extractor.run(img_path, plots_dir, tag=tag)
                extracted_plot_data[fig_number] = result
                print(f"  Successfully extracted data for figure {fig_number}. Saved to: {result['csv_path']}")
            except Exception as e:
                print(f"  Failed to extract data for figure {fig_number}: {e}")

    print(f"  Finished processing data for {len(extracted_plot_data)} plot(s).")

    return {"extracted_plot_data": extracted_plot_data}