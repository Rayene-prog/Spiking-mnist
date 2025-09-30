#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data logs figs
echo "Setup OK. Ensuite: ./scripts/download_mnist.sh"
