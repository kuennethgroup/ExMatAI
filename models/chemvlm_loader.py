"""
ChemVLM Model Loader
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import os

class ChemVLMLoader:
    """Singleton loader for ChemVLM model"""
    
    _instance: Optional['ChemVLMLoader'] = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self.load_model()
    
    def load_model(self, model_id: str = "AI4Chem/ChemVLM-26B-1-2"):
        """Load ChemVLM model and tokenizer"""
        if self._model is not None:
            return
        
        print(f"📥 Loading ChemVLM model: {model_id}")
        
        # Set CUDA launch blocking
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        
        print(f"✅ ChemVLM model loaded successfully")
    
    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load_model()
        return self._tokenizer
    
    def generate_smiles(self, pixel_values: torch.Tensor, 
                       query: str = "Can you tell me what is the molecule in this image, using SMILES format？",
                       gen_kwargs: Optional[dict] = None) -> str:
        """Generate SMILES from image tensor"""
        
        if gen_kwargs is None:
            gen_kwargs = {
                "max_length": 1000,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9
            }
        
        # Ensure pixel values are on correct device
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        
        # Generate response
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            query,
            gen_kwargs
        )
        
        return response
    
    def unload_model(self):
        """Unload model from memory"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ ChemVLM model unloaded")
