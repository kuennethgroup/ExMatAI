"""
MolDetV2 Model Loader
"""

import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from typing import Optional
import os

class MolDetLoader:
    """Singleton loader for MolDetV2 model"""
    
    _instance: Optional['MolDetLoader'] = None
    _model = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self.load_model()
    
    def load_model(self, 
                   repo_id: str = "UniParser/MolDetv2",
                   filename: str = "moldet_v2_yolo11n_640_general.pt"):
        """Load MolDetV2 model"""
        if self._model is not None:
            return
        
        print(f"📥 Loading MolDetV2 model: {repo_id}/{filename}")
        
        # Download model weights
        self._model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model"
        )
        
        # Load YOLO model
        self._model = YOLO(self._model_path)
        
        print(f"✅ MolDetV2 model loaded successfully")
        print(f"   Model path: {self._model_path}")
    
    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model
    
    def predict(self, image_path: str, 
                imgsz: int = 640, 
                conf: float = 0.5,
                device: Optional[int] = None,
                save: bool = False):
        """Run structure detection on image"""
        
        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'
        
        results = self.model.predict(
            image_path,
            save=save,
            imgsz=imgsz,
            conf=conf,
            device=device
        )
        
        return results
    
    def unload_model(self):
        """Unload model from memory"""
        if self._model is not None:
            del self._model
            self._model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ MolDetV2 model unloaded")
