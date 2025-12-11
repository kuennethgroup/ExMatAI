"""
DeepSeek-OCR Wrapper
Executes official DeepSeek-OCR implementation as a black box
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .config_manager import DeepSeekOCRConfigManager

# Get Ollama host from environment
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

class DeepSeekOCRWrapper:
    """Wrapper to execute DeepSeek-OCR in isolated environment"""
    
    def __init__(self, deepseek_ocr_root: str = "./DeepSeek-OCR"):
        self.config_manager = DeepSeekOCRConfigManager(deepseek_ocr_root)
        self.deepseek_ocr_root = Path(deepseek_ocr_root)
    
    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str = "temp/deepseek_ocr_output"
    ) -> Dict:
        """
        Process PDF using DeepSeek-OCR in isolated environment
        
        Args:
            pdf_path: Path to input PDF
            output_dir: Output directory for results
            
        Returns:
            Dict containing:
                - extracted_text: Full text content
                - text_by_page: Dict[page_num, text]
                - images_dir: Directory containing extracted images
                - markdown_file: Path to markdown output
        """
        
        print("\n🔍 Running DeepSeek-OCR (official implementation)...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Write config.py
        print("  ├─ Writing config.py...")
        self.config_manager.write_config(
            input_path=pdf_path,
            output_path=output_dir
        )
        
        # Get paths
        run_script = self.config_manager.get_run_script_path()
        venv_activate = self.config_manager.get_venv_activate_path()
        
        # Prepare command to run in isolated environment
        command = f"""
source {venv_activate} && \
cd {self.config_manager.model_path} && \
python run_dpsk_ocr_pdf.py
"""
        
        print("  ├─ Executing DeepSeek-OCR...")
        print(f"  │  └─ Command: python run_dpsk_ocr_pdf.py")
        
        try:
            # Execute in bash shell
            result = subprocess.run(
                command,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                print(f"  ✗ DeepSeek-OCR failed!")
                print(f"    Error: {result.stderr}")
                raise RuntimeError(f"DeepSeek-OCR failed: {result.stderr}")
            
            print("  ✓ DeepSeek-OCR completed successfully")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("DeepSeek-OCR execution timed out (>10 minutes)")
        
        # Parse outputs
        print("  ├─ Parsing outputs...")
        result = self._parse_outputs(pdf_path, output_dir)
        
        print("  ✓ Output parsing complete")
        return result
    
    def _parse_outputs(self, pdf_path: str, output_dir: str) -> Dict:
        """Parse DeepSeek-OCR outputs"""
        
        pdf_name = Path(pdf_path).stem
        
        # Find markdown file
        markdown_files = list(Path(output_dir).glob(f"{pdf_name}*.mmd"))
        
        if not markdown_files:
            raise FileNotFoundError(f"No markdown output found in {output_dir}")
        
        markdown_file = markdown_files[0]
        
        # Read markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Parse page-by-page content
        text_by_page = self._split_markdown_by_page(markdown_content)
        
        # Get images directory
        images_dir = Path(output_dir) / "images"
        
        return {
            'extracted_text': markdown_content,
            'text_by_page': text_by_page,
            'images_dir': str(images_dir) if images_dir.exists() else None,
            'markdown_file': str(markdown_file),
            'output_dir': output_dir
        }
    
    def _split_markdown_by_page(self, markdown: str) -> Dict[int, str]:
        """Split markdown by page markers"""
        
        # DeepSeek-OCR may include page markers
        # This is a simple implementation - adjust based on actual output format
        
        text_by_page = {}
        current_page = 1
        current_text = []
        
        for line in markdown.split('\n'):
            # Check for page markers (adjust pattern based on actual output)
            if line.strip().startswith('---') or 'PAGE' in line.upper():
                # Save current page
                if current_text:
                    text_by_page[current_page] = '\n'.join(current_text)
                    current_text = []
                    current_page += 1
            else:
                current_text.append(line)
        
        # Save last page
        if current_text:
            text_by_page[current_page] = '\n'.join(current_text)
        
        # If no page markers found, treat as single page
        if not text_by_page:
            text_by_page[1] = markdown
        
        return text_by_page
    
    def cleanup(self, output_dir: str):
        """Cleanup temporary outputs"""
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
