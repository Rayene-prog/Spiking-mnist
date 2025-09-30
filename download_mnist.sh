#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
cd data
echo "📥 Téléchargement MNIST..."
wget -q http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -q http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -q http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -q http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip -f *.gz
echo " Données prêtes dans ./data"
