#!/bin/bash
# setup.sh - Initial setup for vast.ai

echo "GPAD Training Setup"
echo "==================="

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "Warning: No GPU detected"
fi

# Install dependencies
echo "Installing dependencies..."
apt-get update -qq && apt-get install -y wget git vim tmux -qq
pip install --upgrade pip -q
pip install -r requirements_gpad2.txt -q

# Create base directories
mkdir -p ./data ./cache

echo ""
echo "Setup complete!"
echo "Next: Upload data to ./data/ then run ./gpad_train.sh"