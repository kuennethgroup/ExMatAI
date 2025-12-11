"""
DeepSeek-OCR Config Manager
Dynamically rewrites config.py without modifying run_dpsk_ocr_pdf.py
"""

import os
from pathlib import Path
from typing import Optional

class DeepSeekOCRConfigManager:
    """Manages DeepSeek-OCR config.py file dynamically"""
    
    def __init__(self, deepseek_ocr_root: str = "./DeepSeek-OCR"):
        self.deepseek_ocr_root = Path(deepseek_ocr_root)
        self.config_path = self.deepseek_ocr_root / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm" / "config.py"
        self.model_path = self.deepseek_ocr_root / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
    
    def write_config(
        self,
        input_path: str,
        output_path: str,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        min_crops: int = 2,
        max_crops: int = 6,
        max_concurrency: int = 100,
        num_workers: int = 64
    ) -> str:
        """
        Write config.py with specified parameters
        
        Args:
            input_path: Path to input PDF file
            output_path: Path to output directory
            model_path: HuggingFace model path or local path
            
        Returns:
            Path to written config file
        """
        
        # Convert to absolute paths
        input_path = str(Path(input_path).absolute())
        output_path = str(Path(output_path).absolute())
        
        # Create output directory if not exists
        os.makedirs(output_path, exist_ok=True)
        
        # Config template
        config_content = f'''
BASE_SIZE = {base_size}
IMAGE_SIZE = {image_size}
CROP_MODE = {crop_mode}
MIN_CROPS = {min_crops}
MAX_CROPS = {max_crops}  # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = {max_concurrency}  # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = {num_workers}  # image pre-process (resize/padding) workers 
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = '{model_path}'  # change to your model path

# Input/Output paths (dynamically set by ExMatAI)
INPUT_PATH = '{input_path}'  # PDF file path
OUTPUT_PATH = '{output_path}'  # Output directory

PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'

from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
'''
        
        # Write config
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"✓ Config written to: {self.config_path}")
        print(f"  ├─ Input: {input_path}")
        print(f"  └─ Output: {output_path}")
        
        return str(self.config_path)
    
    def get_run_script_path(self) -> str:
        """Get path to run_dpsk_ocr_pdf.py"""
        return str(self.model_path / "run_dpsk_ocr_pdf.py")
    
    def get_venv_activate_path(self) -> str:
        """Get path to DeepSeek-OCR venv activation script"""
        return str(self.deepseek_ocr_root / ".venv" / "bin" / "activate")
