#!/usr/bin/env bash
# One-time setup: creates a virtual environment and installs all dependencies.
# Run this once before using the repo.
#
# Usage:
#   bash setup.sh

set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install numpy scipy brian2

echo ""
echo "Setup complete."
echo ""
echo "To generate the simulation data:"
echo "  source venv/bin/activate"
echo "  python simulation/generate_brian2_hdmea_burst.py"
echo ""
echo "To run validation methods:"
echo "  python validation/glmcc_validate.py"
echo "  python validation/dsttc_sccg_validate.py sccg"
echo "  python validation/dsttc_sccg_validate.py dsttc"
