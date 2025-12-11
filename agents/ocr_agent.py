"""
OCR Agent - Extract text and images from PDF using DeepSeek-OCR (Official Implementation)
"""

import os
import ollama
import fitz  # PyMuPDF
from PIL import Image
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv
import io

# Load environment
load_dotenv()

from utils.state_schema import AgentState
from utils.deepseek_ocr_wrapper import DeepSeekOCRWrapper
from utils.text_processing import extract_structure_labels, extract_figure_caption

# Configure Ollama client
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

def ocr_agent(state: AgentState) -> AgentState:
    """
    Extract text and images from PDF using DeepSeek-OCR official implementation
    """
    print("\n" + "="*80)
    print("🔍 AGENT 1: OCR & Document Analysis (DeepSeek-OCR)")
    print("="*80)
    
    pdf_path = state['pdf_path']
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Initialize DeepSeek-OCR wrapper
    print("\n📦 Initializing DeepSeek-OCR wrapper...")
    ocr_wrapper = DeepSeekOCRWrapper()
    
    # Process PDF using official implementation
    try:
        ocr_result = ocr_wrapper.process_pdf(
            pdf_path=pdf_path,
            output_dir="temp/deepseek_ocr_output"
        )
        
        extracted_text = ocr_result['extracted_text']
        text_by_page = ocr_result['text_by_page']
        
        print(f"\n✓ OCR Complete:")
        print(f"  ├─ Pages processed: {len(text_by_page)}")
        print(f"  ├─ Total text: {len(extracted_text)} characters")
        print(f"  └─ Output: {ocr_result['output_dir']}")
        
    except Exception as e:
        print(f"\n✗ DeepSeek-OCR failed: {e}")
        raise
    
    # Now process images for structure/plot detection using PyMuPDF
    print(f"\n📄 Processing images for structure/plot detection...")
    
    # Open PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    
    structure_images = []
    plot_images = []
    
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    
    # Use Qwen3-VL for image classification
    # Configure client with Docker Ollama
    client = ollama.Client(host=OLLAMA_HOST)
    
    for page_num in range(num_pages):
        print(f"  ├─ Analyzing page {page_num + 1}/{num_pages}...")
        
        # Get page
        page = doc[page_num]
        
        # Render page to image (300 DPI)
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI scaling
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        page_image = Image.open(io.BytesIO(img_data))
        
        # Save image
        page_path = f"temp/page_{page_num + 1}.png"
        page_image.save(page_path, 'PNG')
        
        page_text = text_by_page.get(page_num + 1, "")
        
        # Analyze image content using Qwen3-VL
        try:
            analysis = client.generate(
                model='qwen3-vl:8b',
                prompt=f"Analyze this page and identify if it contains: 1) Molecular/chemical structures, 2) Plots/graphs, 3) Tables. List what you find.",
                images=[page_path]
            )
            
            analysis_text = analysis['response'].lower()
            
            # Check for structures
            if any(kw in analysis_text for kw in ['structure', 'molecule', 'chemical', 'compound']):
                nearby_text = extract_structure_labels(page_text, analysis['response'])
                
                structure_images.append({
                    'page': page_num + 1,
                    'image': page_image,
                    'image_path': page_path,
                    'nearby_text': nearby_text,
                    'analysis': analysis['response']
                })
                print(f"  │  ✓ Found structure region")
            
            # Check for plots
            if any(kw in analysis_text for kw in ['plot', 'graph', 'figure', 'chart']):
                caption = extract_figure_caption(page_text, page_num + 1)
                
                plot_images.append({
                    'page': page_num + 1,
                    'image': page_image,
                    'image_path': page_path,
                    'caption': caption,
                    'analysis': analysis['response']
                })
                print(f"  │  ✓ Found plot")
        
        except Exception as e:
            print(f"  │  ⚠️  Image analysis failed: {e}")
            # Continue processing other pages
    
    # Close PDF document
    doc.close()
    
    print(f"\n" + "="*80)
    print(f"✅ OCR Complete:")
    print(f"  ├─ Pages processed: {len(text_by_page)}")
    print(f"  ├─ Total text: {len(extracted_text)} characters")
    print(f"  ├─ Structure images: {len(structure_images)}")
    print(f"  └─ Plot images: {len(plot_images)}")
    print("="*80)
    
    return {
        **state,
        "extracted_text": extracted_text,
        "text_by_page": text_by_page,
        "structure_images": structure_images,
        "plot_images": plot_images,
        "messages": state.get('messages', []) + ["OCR completed"],
        "current_agent": "text_analysis"
    }
