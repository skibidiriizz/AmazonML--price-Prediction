#!/bin/bash

# Setup script for CLIP-based Price Prediction Pipeline
echo "🚀 Setting up CLIP-based Price Prediction Environment"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (CPU version - change if you need GPU)
echo "🔥 Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📋 Installing other requirements..."
pip install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python3 -c "import clip; print('✓ CLIP: Available')"
python3 -c "import pandas as pd; print(f'✓ Pandas: {pd.__version__}')"
python3 -c "from PIL import Image; print('✓ Pillow: Available')"
python3 -c "import requests; print('✓ Requests: Available')"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the pipeline:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the pipeline: python main.py"
echo ""
echo "For help: python main.py --help"