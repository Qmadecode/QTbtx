#!/bin/bash
# QTbtx Google Compute Engine Setup Script
# Run this ON the GCE VM after SSH

echo "============================================="
echo "QTbtx GCE Setup"
echo "============================================="

# Update system
echo "Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git

# Clone repo
echo "Cloning QTbtx..."
git clone https://github.com/Qmadecode/QTbtx.git
cd QTbtx

# Create virtual environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p data/preprocessed
mkdir -p results

echo ""
echo "============================================="
echo "Setup Complete!"
echo "============================================="
echo ""
echo "Next steps:"
echo "1. Upload your data:"
echo "   gsutil -m cp -r gs://YOUR_BUCKET/preprocessed/* data/preprocessed/"
echo ""
echo "2. Run optimization:"
echo "   source venv/bin/activate"
echo "   python run_optimizer.py --signal all --data data/preprocessed"
echo ""
echo "3. Download results:"
echo "   gsutil -m cp results/*.xlsx gs://YOUR_BUCKET/results/"

