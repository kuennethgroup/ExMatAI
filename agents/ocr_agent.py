"""
OCR Agent - DeepSeek-OCR Subprocess Node
Rewrites config.py then runs the official run_dpsk_ocr_pdf.py in the
isolated DeepSeek-OCR venv. Returns the .mmd path and images directory.
"""

import os
import re
import subprocess
from pathlib import Path

from utils.state_schema import WorkflowState


def run_deepseek_ocr(state: WorkflowState) -> WorkflowState:
    """LangGraph node: run DeepSeek-OCR on the input PDF."""
    print("\n" + "=" * 70)
    print("Step 1: Running DeepSeek-OCR")
    print("=" * 70)

    pdf_path = os.path.abspath(state["pdf_path"])
    output_dir = os.path.abspath(state["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # 1. Paths -----------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "DeepSeek-OCR" / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm" / "config.py"
    script_path = project_root / "DeepSeek-OCR" / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm" / "run_dpsk_ocr_pdf.py"
    venv_python = project_root / "DeepSeek-OCR" / ".venv" / "bin" / "python"

    # 2. Rewrite config.py dynamically -----------------------------------
    print(f"  Updating config for input PDF: {pdf_path}")
    print(f"  Output directory set to: {output_dir}")

    with open(config_path, "r") as f:
        config_content = f.read()

    config_content = re.sub(
        r"INPUT_PATH\s*=\s*['\"].*?['\"]",
        f"INPUT_PATH='{pdf_path}'",
        config_content,
    )
    config_content = re.sub(
        r"OUTPUT_PATH\s*=\s*['\"].*?['\"]",
        f"OUTPUT_PATH='{output_dir}'",
        config_content,
    )

    with open(config_path, "w") as f:
        f.write(config_content)

    # 3. Unload Ollama models to free GPU for DeepSeek-OCR ---------------
    try:
        import requests
        ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ps_resp = requests.get(f"{ollama_url}/api/ps", timeout=5)
        if ps_resp.status_code == 200:
            loaded = ps_resp.json().get("models", [])
            if loaded:
                print(f"  Unloading {len(loaded)} Ollama model(s) to free up the GPU...")
                for m in loaded:
                    requests.post(
                        f"{ollama_url}/api/generate",
                        json={"model": m["name"], "keep_alive": 0},
                        timeout=10,
                    )
    except Exception:
        pass  # Non-critical: if Ollama isn't running, DeepSeek-OCR just runs

    # 4. Run DeepSeek-OCR subprocess -------------------------------------
    print("  Starting the DeepSeek-OCR process. This might take a moment...")
    result = subprocess.run(
        [str(venv_python), str(script_path)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=1200,  # 20-minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"DeepSeek-OCR failed:\n{result.stderr}")

    print("  DeepSeek-OCR finished successfully.")

    # 5. Locate outputs --------------------------------------------------
    pdf_name = Path(pdf_path).stem
    mmd_path = os.path.join(output_dir, f"{pdf_name}.mmd")
    images_dir = os.path.join(output_dir, "images")

    if not os.path.exists(mmd_path):
        # Try to find it with glob
        mmd_files = list(Path(output_dir).glob("*.mmd"))
        if mmd_files:
            mmd_path = str(mmd_files[0])
        else:
            raise FileNotFoundError(f"No .mmd file found in {output_dir}")

    print(f"  Located MMD file: {mmd_path}")
    print(f"  Images are stored in: {images_dir}")

    return {"mmd_path": mmd_path, "images_dir": images_dir}