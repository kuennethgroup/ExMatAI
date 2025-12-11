"""
ExMat AI Setup Script
Installs dependencies and downloads required models
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run shell command with error handling"""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    try:
        subprocess.run(cmd, check=True, shell=isinstance(cmd, str))
        print(f"✅ {description} - Success")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        return False
    return True

def setup_environment():
    """Complete environment setup"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                    ExMat AI Setup Script                       ║
    ║          Automated Battery Material Data Extraction            ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required. Current:", sys.version)
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
    except ImportError:
        print("ℹ️  PyTorch not yet installed")
    
    # Install core dependencies
    if not run_command(
        ["uv", "pip", "install", "-r", "requirements.txt"],
        "Installing Python dependencies"
    ):
        return False
    
    # Check Ollama installation
    print(f"\n{'='*80}")
    print("🔍 Checking Ollama installation")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"✅ Ollama installed: {result.stdout.strip()}")
        
        # Pull Ollama models
        ollama_models = [
            ("deepseek-ocr", "DeepSeek-OCR (6.7GB)"),
            ("qwen3:8b", "Qwen3-8B (4.7GB)"),
            ("qwen3-vl:8b", "Qwen3-VL-8b (6.1GB)")
        ]
        
        for model, desc in ollama_models:
            if not run_command(
                f"ollama pull {model}",
                f"Downloading {desc}"
            ):
                print(f"⚠️  Failed to pull {model}")
    
    except FileNotFoundError:
        print("❌ Ollama not found. Please install from: https://ollama.com/download")
        return False
    
    # Create necessary directories
    print(f"\n{'='*80}")
    print("📁 Creating project directories")
    print(f"{'='*80}")
    
    directories = ["outputs", "logs", "temp"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✓ {dir_name}/")
    
    # Download MolDetV2 weights
    print(f"\n{'='*80}")
    print("📥 Pre-downloading MolDetV2 model weights")
    print(f"{'='*80}")
    
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="UniParser/MolDetv2",
            filename="moldet_v2_yolo11n_640_general.pt",
            repo_type="model"
        )
        print(f"✅ MolDetV2 downloaded to: {model_path}")
    except Exception as e:
        print(f"⚠️  MolDetV2 download failed: {e}")
    
    # Create .env file if not exists
    if not Path(".env").exists():
        print(f"\n{'='*80}")
        print("📝 Creating .env configuration file")
        print(f"{'='*80}")
        with open(".env", "w") as f:
            f.write("""# ExMat AI Configuration
# GPU Device
CUDA_VISIBLE_DEVICES=0

# Model Settings
CHEMVLM_MAX_NUM=6
MOLDET_CONFIDENCE=0.5
QWEN3_TEMPERATURE=0.7

# Processing
BATCH_SIZE=1
MAX_WORKERS=4

# Logging
LOG_LEVEL=INFO
""")
        print("✅ Created .env file (edit as needed)")
    
    # Run basic test
    print(f"\n{'='*80}")
    print("🧪 Running basic system test")
    print(f"{'='*80}")
    
    try:
        test_code = """
import torch
import transformers
from transformers import AutoTokenizer
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ CUDA:', torch.cuda.is_available())
"""
        subprocess.run(["uv", "run", "-"], 
            input=test_code,
            text=True,
            check=True
        )
    except Exception as e:
        print(f"⚠️  System test warning: {e}")
    
    # Setup complete
    print(f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║                  ✅ Setup Complete!                             ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Next steps:
    1. Review configuration in .env
    2. Run: python main.py --pdf sample_data_paper.pdf
    3. Check outputs/ folder for results
    
    For help: python main.py --help
    """)
    
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
