"""
Structure Extraction Agent - Detect chemical structures using MolDetV2 and label them
"""

import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image
import ollama
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from utils.state_schema import AgentState

# Configure Ollama client
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
client = ollama.Client(host=OLLAMA_HOST)

def structure_extraction_agent(state: AgentState) -> AgentState:
    """
    Detect chemical structures using MolDetV2 and label them with qwen3-vl
    """
    print("\n" + "="*80)
    print("🔬 AGENT 3: Chemical Structure Detection & Labeling")
    print("="*80)
    
    structure_images = state.get('structure_images', [])
    
    if not structure_images:
        print("\n⚠️  No structure images to process - skipping")
        return {
            **state,
            "detected_structures": [],
            "messages": state.get('messages', []) + ["No structures to detect"],
            "current_agent": "smiles_generation"
        }
    
    print(f"\n📊 Processing {len(structure_images)} images with MolDetV2...")
    
    # Load MolDetV2 model (auto-downloads from HuggingFace)
    try:
        print(f"  ├─ Loading MolDetV2 model...")
        
        model_path = hf_hub_download(
            repo_id="UniParser/MolDetv2",
            filename="moldet_v2_yolo11n_640_general.pt",
            repo_type="model",
            cache_dir="models/moldetv2"
        )
        
        print(f"  │  ├─ Model cached at: {model_path}")
        
        model = YOLO(model_path)
        print(f"  ✓ MolDetV2 loaded successfully")
        
    except Exception as e:
        print(f"  ✗ Failed to load MolDetV2: {e}")
        return {
            **state,
            "detected_structures": [],
            "errors": state.get('errors', []) + [f"MolDetV2 loading failed: {e}"],
            "messages": state.get('messages', []) + ["Structure detection failed"],
            "current_agent": "smiles_generation"
        }
    
    detected_structures = []
    
    for idx, struct_img in enumerate(structure_images, 1):
        print(f"\n  ├─ Processing image {idx}/{len(structure_images)} (page {struct_img['page']})...")
        
        try:
            # Save image temporarily
            temp_path = f"temp/struct_detect_p{struct_img['page']}_{idx}.png"
            struct_img['image'].save(temp_path)
            
            # Run MolDetV2 detection
            results = model.predict(
                temp_path,
                save=False,
                imgsz=640,
                conf=0.5,
                device=0  # GPU
            )
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                print(f"  │  ✓ Found {len(results[0].boxes)} structure(s)")
                
                # Process each detected structure
                for box_idx, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Crop structure
                    structure_crop = struct_img['image'].crop((int(x1), int(y1), int(x2), int(y2)))
                    
                    # Expand bbox downward to capture label
                    img_width, img_height = struct_img['image'].size
                    label_y1 = int(y2)
                    label_y2 = min(int(y2) + 150, img_height)  # Expand 150px down
                    label_region = struct_img['image'].crop((int(x1), label_y1, int(x2), label_y2))
                    
                    # Save crops
                    crop_path = f"temp/structure_p{struct_img['page']}_{idx}_{box_idx}.png"
                    structure_crop.save(crop_path)
                    
                    label_path = f"temp/label_region_p{struct_img['page']}_{idx}_{box_idx}.png"
                    label_region.save(label_path)
                    
                    # Use qwen3-vl to extract label text
                    try:
                        label_response = client.generate(
                            model='qwen3-vl:8b',
                            prompt="What is the chemical compound name or label shown in this image? Return only the name/label, nothing else.",
                            images=[label_path]
                        )
                        
                        material_name = label_response['response'].strip()
                        
                        # If no label found below, check near the structure
                        if not material_name or len(material_name) < 2:
                            # Expand bbox in all directions
                            expand_x1 = max(0, int(x1) - 100)
                            expand_y1 = max(0, int(y1) - 100)
                            expand_x2 = min(img_width, int(x2) + 100)
                            expand_y2 = min(img_height, int(y2) + 100)
                            
                            near_region = struct_img['image'].crop((expand_x1, expand_y1, expand_x2, expand_y2))
                            near_path = f"temp/near_region_p{struct_img['page']}_{idx}_{box_idx}.png"
                            near_region.save(near_path)
                            
                            near_response = client.generate(
                                model='qwen3-vl:8b',
                                prompt="What is the chemical compound name or label near this structure? Return only the name/label.",
                                images=[near_path]
                            )
                            
                            material_name = near_response['response'].strip()
                        
                        print(f"  │    • Structure {box_idx + 1}: '{material_name}' (conf: {float(box.conf[0]):.2f})")
                        
                    except Exception as e:
                        print(f"  │    ⚠️  Label extraction failed: {e}")
                        material_name = f"Unknown_{struct_img['page']}_{box_idx}"
                    
                    detected_structures.append({
                        'page': struct_img['page'],
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(box.conf[0]),
                        'image_path': crop_path,
                        'image': structure_crop,
                        'material_name': material_name,
                        'nearby_text': struct_img.get('nearby_text', '')
                    })
            else:
                print(f"  │  ⚠️  No structures detected")
                
        except Exception as e:
            print(f"  │  ✗ Detection failed: {e}")
            state['errors'].append(f"Structure detection failed for page {struct_img['page']}: {e}")
    
    print(f"\n" + "="*80)
    print(f"✅ Structure Detection Complete:")
    print(f"  ├─ Images processed: {len(structure_images)}")
    print(f"  └─ Structures detected & labeled: {len(detected_structures)}")
    print("="*80)
    
    return {
        **state,
        "detected_structures": detected_structures,
        "messages": state.get('messages', []) + ["Structure detection completed"],
        "current_agent": "smiles_generation"
    }
