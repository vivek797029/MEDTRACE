#!/bin/bash
# MedTrace Setup Script for Mac M2
# Run this FIRST: chmod +x setup.sh && ./setup.sh

echo "=========================================="
echo "  MedTrace - Medical Reasoning Chains"
echo "  Setup for Apple Silicon (M2)"
echo "=========================================="

# Check if Python 3.10+ exists
python3 --version 2>/dev/null || { echo "❌ Python 3 not found. Install from python.org"; exit 1; }

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv medtrace_env
source medtrace_env/bin/activate

# Install PyTorch with MPS support (Apple Silicon)
echo "🔥 Installing PyTorch with MPS (Metal) support..."
pip install --upgrade pip
pip install torch torchvision torchaudio

# Install ML dependencies
echo "📚 Installing ML dependencies..."
pip install transformers>=4.36.0
pip install peft>=0.7.0
pip install datasets>=2.16.0
pip install accelerate>=0.25.0
pip install sentencepiece>=0.1.99
pip install protobuf>=3.20.0
pip install tqdm pandas scikit-learn

# Verify MPS is available
echo ""
echo "🔍 Checking MPS (Metal GPU) availability..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Metal) is available! Your M2 GPU will accelerate training.')
    print(f'   PyTorch version: {torch.__version__}')
else:
    print('⚠️  MPS not available. Training will use CPU (slower but still works).')
"

echo ""
echo "=========================================="
echo "  ✅ Setup complete!"
echo "  To activate: source medtrace_env/bin/activate"
echo "  Next step:   python src/01_download_data.py"
echo "=========================================="
