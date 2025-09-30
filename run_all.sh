#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python snn.py          # entraîne, produit logs/ (+ figs/ si activé)
python viz_extra.py    # génère toutes les figures
python eval_rbf.py || echo "ℹ️ eval_rbf.py (optionnel)"
echo "✅ Terminé. Voir ./logs et ./figs"
