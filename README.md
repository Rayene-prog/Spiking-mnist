# Spiking-mnist
Réseau convolutionnel à spikes (SNN) avec apprentissage STDP pour MNIST. Encodage en spikes (DoG + temps), entraînement non supervisé couche par couche et classification via SVM linéaire (MAX/SUM). Scripts fournis pour analyse et visualisation des résultats.

Ce projet utilise le jeu MNIST (70 000 chiffres manuscrits, 28×28).

Il faut disposer des 4 fichiers binaires au format idx-ubyte :

train-images-idx3-ubyte

train-labels-idx1-ubyte

t10k-images-idx3-ubyte

t10k-labels-idx1-ubyte

 Téléchargement officiel : http://yann.lecun.com/exdb/mnist/

Installation (Linux / macOS)

Dans un terminal, placez-vous dans le dossier data/ du projet puis lancez :

cd SpikingConvNet-main/data

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz


Après décompression, vous devez obtenir cette structure :

SpikingConvNet-main/

│── snn.py

│── utils.py

│── viz_extra.py

│── eval_rbf.py

│── data/

│   ├── train-images-idx3-ubyte

│   ├── train-labels-idx1-ubyte

│   ├── t10k-images-idx3-ubyte

│   └── t10k-labels-idx1-ubyte


