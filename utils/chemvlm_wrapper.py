"""
ChemVLM Wrapper - Generate SMILES from structure images
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import os
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=6):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class ChemVLMWrapper:
    """Wrapper for ChemVLM SMILES generation"""
    
    def __init__(self, model_name="AI4Chem/ChemVLM-26B-1-2", cache_dir="models/chemvlm"):
        """
        Initialize ChemVLM model
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Local cache directory
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load ChemVLM model (auto-downloads if not cached)"""
        
        print(f"  ├─ Loading ChemVLM from {self.model_name}...")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load tokenizer and model (auto-downloads to cache)
            print(f"  │  ├─ Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir)
            )
            
            print(f"  │  ├─ Loading model (this may take 10-20 minutes on first run)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
                cache_dir=str(self.cache_dir)
            ).eval()
            
            print(f"  ✓ ChemVLM loaded successfully")
            return True
            
        except Exception as e:
            print(f"  ✗ Failed to load ChemVLM: {e}")
            return False
    
    def generate_smiles(self, image_path_or_pil, query=None):
        """
        Generate SMILES from structure image
        
        Args:
            image_path_or_pil: Path to image or PIL Image
            query: Custom query (optional)
            
        Returns:
            str: SMILES string
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if query is None:
            query = "Can you tell me what is the molecule in this image, using SMILES format?"
        
        # Load and preprocess image
        pixel_values = load_image(image_path_or_pil, max_num=6).to(torch.bfloat16).cuda()
        
        # Generate SMILES
        gen_kwargs = {
            "max_length": 1000,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        response = self.model.chat(self.tokenizer, pixel_values, query, gen_kwargs)
        
        return response.strip()
