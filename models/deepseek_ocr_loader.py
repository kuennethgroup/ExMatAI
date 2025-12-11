"""
DeepSeek-OCR Model Loader for vLLM (High Performance)
"""

import torch
from vllm import LLM, SamplingParams
from typing import Optional, Dict, List
import os

class DeepSeekOCRVLLMLoader:
    """Singleton loader for DeepSeek-OCR with vLLM"""
    
    _instance: Optional['DeepSeekOCRVLLMLoader'] = None
    _llm = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._llm is None:
            self.load_model()
    
    def load_model(self, model_id: str = "deepseek-ai/DeepSeek-OCR"):
        """Load DeepSeek-OCR model with vLLM"""
        if self._llm is not None:
            return
        
        print(f"📥 Loading DeepSeek-OCR model with vLLM: {model_id}")
        
        # Import vLLM-specific processor
        try:
            from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
        except ImportError:
            print("⚠️  NGramPerReqLogitsProcessor not found - using default")
            NGramPerReqLogitsProcessor = None
        
        # Initialize vLLM
        self._llm = LLM(
            model=model_id,
            trust_remote_code=True,
            enable_prefix_caching=False,
            logits_processors=[NGramPerReqLogitsProcessor] if NGramPerReqLogitsProcessor else None,
            max_model_len=8192,
            gpu_memory_utilization=0.8
        )
        
        print(f"✅ DeepSeek-OCR model loaded successfully with vLLM")
    
    @property
    def llm(self):
        if self._llm is None:
            self.load_model()
        return self._llm
    
    def extract_text(self, 
                     image_path: str,
                     prompt: str = "<image>\n<|grounding|>Convert the document to markdown.") -> str:
        """
        Extract text from image using vLLM
        
        Args:
            image_path: Path to image file
            prompt: Prompt for OCR task
        
        Returns:
            Extracted text
        """
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822}  # Table tokens
            ) if hasattr(self, '_use_ngram') else {}
        )
        
        # Create input
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_path
            }
        }
        
        try:
            # Generate
            outputs = self.llm.generate([inputs], sampling_params)
            
            if outputs and len(outputs) > 0:
                return outputs[0].outputs[0].text
            else:
                raise ValueError("No output generated")
        
        except Exception as e:
            print(f"❌ vLLM inference failed: {e}")
            raise
    
    def batch_extract_text(self,
                          image_paths: List[str],
                          prompts: Optional[List[str]] = None) -> List[str]:
        """
        Batch extract text from multiple images (much faster)
        
        Args:
            image_paths: List of image paths
            prompts: Optional list of prompts (one per image)
        
        Returns:
            List of extracted texts
        """
        
        if prompts is None:
            prompts = ["<image>\n<|grounding|>Convert the document to markdown."] * len(image_paths)
        
        # Create batch inputs
        inputs = []
        for image_path, prompt in zip(image_paths, prompts):
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_path
                }
            })
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192
        )
        
        # Batch generate
        outputs = self.llm.generate(inputs, sampling_params)
        
        # Extract texts
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text)
            else:
                results.append("")
        
        return results
    
    def unload_model(self):
        """Unload model from memory"""
        if self._llm is not None:
            del self._llm
            self._llm = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ DeepSeek-OCR (vLLM) model unloaded")
