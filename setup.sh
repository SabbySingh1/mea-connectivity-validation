#!/usr/bin/env bash
# One-time setup: creates a virtual environment and installs all dependencies.
# Run this once before using the repo.
#
# Usage:
#   bash setup.sh
#
# Note: use a Python 3.9+ interpreter that can actually build these packages.
# On macOS, the system `python3` may not have pip/venv working correctly —
# if `python3 -m venv venv` fails, try the Python bundled with Xcode
# Command Line Tools instead, e.g.:
#   /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9 -m venv venv

set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo ""
echo "trial_157 (the primary validated benchmark) is already included at:"
echo "  data/trial_157_spikes.npz"
echo "  data/trial_157_connectivity.npz"
echo ""
echo "To reproduce the validated sCCG result (F1=0.460) on trial_157:"
echo "  source venv/bin/activate"
echo "  python validation/dsttc_sccg_validate.py sccg \\"
echo "      --spikes data/trial_157_spikes.npz \\"
echo "      --conn   data/trial_157_connectivity.npz \\"
echo "      --alpha  0.001"
echo ""
echo "To run GLMCC on trial_157:"
echo "  python validation/glmcc_validate.py \\"
echo "      --spikes data/trial_157_spikes.npz \\"
echo "      --conn   data/trial_157_connectivity.npz"
echo ""
echo "To generate the older burst-simulation dataset instead:"
echo "  python simulation/generate_brian2_hdmea_burst.py"
