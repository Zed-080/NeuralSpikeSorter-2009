#!/bin/bash
echo "=========================================="
echo "Setting up Neural Spike Sorter Environment"
echo "=========================================="

# 1. Check Python
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] python3 could not be found."
    exit 1
fi

# 2. Create Venv
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[INFO] Virtual environment already exists."
fi

# 3. Install
echo "[INFO] Installing requirements..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "To run the pipeline, use:"
echo "source .venv/bin/activate && python main.py"
echo "=========================================="