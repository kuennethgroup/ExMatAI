#!/bin/bash

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║            ExMat AI Environment Setup Script                   ║"
echo "║          Setting up DeepSeek-OCR + ExMatAI environments        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✓ uv is installed: $(uv --version)"
echo ""

# ============================================================================
# PART 1: Setup DeepSeek-OCR Environment (ISOLATED)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 PART 1: Setting up DeepSeek-OCR Environment (ISOLATED)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Clone DeepSeek-OCR if not exists
if [ ! -d "DeepSeek-OCR" ]; then
    echo "📥 Cloning DeepSeek-OCR repository..."
    git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
    echo "✓ Repository cloned"
else
    echo "✓ DeepSeek-OCR directory already exists"
fi

cd DeepSeek-OCR

# Remove any existing venv
if [ -d ".venv" ]; then
    echo "🗑️  Removing existing .venv..."
    rm -rf .venv
fi

# Create isolated virtual environment
echo ""
echo "🐍 Creating ISOLATED DeepSeek-OCR virtual environment (Python 3.10)..."

UV_PROJECT_ENVIRONMENT="" uv venv .venv --python 3.10 --seed

echo "✓ Virtual environment created at DeepSeek-OCR/.venv"

# Activate environment
source .venv/bin/activate

echo ""
echo "📦 Installing DeepSeek-OCR dependencies..."

# Install setuptools first (needed for flash-attn)
echo "  ├─ Installing setuptools..."
uv pip install setuptools wheel

# Install PyTorch with CUDA 11.8
echo "  ├─ Installing PyTorch 2.6.0 (CUDA 11.8)..."
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Check if vllm wheel exists
if [ -f "../resources/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl" ]; then
    echo "  ├─ Installing vLLM from local wheel..."
    uv pip install ../resources/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
else
    echo "  ├─ Installing vLLM from PyPI..."
    uv pip install vllm==0.8.5
fi

# Install official requirements
echo "  ├─ Installing official requirements..."
cat > requirements_deepseek_temp.txt << 'EOF'
transformers==4.46.3
tokenizers==0.20.3
PyMuPDF
img2pdf
einops
easydict
addict
Pillow
numpy
EOF

uv pip install -r requirements_deepseek_temp.txt
rm requirements_deepseek_temp.txt

# Install flash-attention with setuptools available
echo "  ├─ Installing flash-attention (this may take 5-10 minutes)..."
uv pip install flash-attn==2.7.3 --no-build-isolation

deactivate

echo ""
echo "✅ DeepSeek-OCR environment setup complete!"
echo "   Location: $(pwd)/.venv"
echo "   Python: $($(pwd)/.venv/bin/python --version)"
echo ""

cd ..

# ============================================================================
# PART 2: Setup Main ExMatAI Environment
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 PART 2: Setting up ExMatAI Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create ExMatAI virtual environment
if [ -d ".venv" ]; then
    echo "🗑️  Removing existing .venv..."
    rm -rf .venv
fi

echo "🐍 Creating ExMatAI virtual environment (Python 3.11)..."
uv venv .venv --python 3.11 --seed

echo "✓ Virtual environment created at .venv"

# Activate environment
source .venv/bin/activate

echo ""
echo "📦 Installing ExMatAI dependencies..."

# Install dependencies from requirements.txt
echo "  ├─ Installing core requirements..."
uv pip install -r requirements.txt

# Install MolNexTR (SMILES generation from molecular structure images)
echo "  ├─ Installing MolNexTR (from GitHub)..."
uv pip install git+https://github.com/CYF2000127/MolNexTR

deactivate

echo ""
echo "✅ ExMatAI environment setup complete!"
echo "   Location: $(pwd)/.venv"
echo "   Python: $(.venv/bin/python --version)"
echo ""


# ============================================================================
# Create .uvignore to prevent workspace detection
# ============================================================================

echo "📝 Creating .uvignore to isolate DeepSeek-OCR..."
cat > .uvignore << 'EOF'
# Ignore DeepSeek-OCR from uv workspace detection
DeepSeek-OCR/
DeepSeek-OCR/**
EOF

echo "✓ Created .uvignore"
echo ""

# ============================================================================
# Final Instructions
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL ENVIRONMENTS SETUP COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Environment Summary:"
echo ""
echo "  DeepSeek-OCR Environment:"
echo "    Path: ./DeepSeek-OCR/.venv"
echo "    Python: 3.10"
echo "    Packages: torch, vllm, transformers, flash-attn"
echo ""
echo "  ExMatAI Environment:"
echo "    Path: ./.venv"
echo "    Python: 3.11"
echo "    Packages: langgraph, ultralytics, figpanel, MolNexTR, langchain-ollama, etc."
echo ""
echo "📝 Usage:"
echo ""
echo "  To use ExMatAI:"
echo "    source .venv/bin/activate"
echo "    python main.py --pdf your_paper.pdf"
echo ""
echo "  To test DeepSeek-OCR separately:"
echo "    cd DeepSeek-OCR"
echo "    source .venv/bin/activate"
echo "    cd DeepSeek-OCR-master/DeepSeek-OCR-vllm"
echo "    python run_dpsk_ocr_pdf.py"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
