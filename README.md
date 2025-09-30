# Spiking-mnist
Réseau convolutionnel à spikes (SNN) avec apprentissage STDP pour MNIST. Encodage en spikes (DoG + temps), entraînement non supervisé couche par couche et classification via SVM linéaire (MAX/SUM). Scripts fournis pour analyse et visualisation des résultats.

📊 Données d’entrée

Ce projet utilise le jeu MNIST (70 000 chiffres manuscrits, 28×28 pixels).
Pour exécuter les scripts, vous devez disposer des 4 fichiers binaires au format idx-ubyte :

train-images-idx3-ubyte

train-labels-idx1-ubyte

t10k-images-idx3-ubyte

t10k-labels-idx1-ubyte

🔹 Téléchargement des données
Option 1 : Téléchargement manuel (navigateur)

Les fichiers sont disponibles sur la page officielle :
👉 http://yann.lecun.com/exdb/mnist/

Téléchargez les 4 fichiers suivants :

train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz

Décompressez-les (clic droit → “Extraire ici” ou via un outil comme 7-Zip/WinRAR) afin d’obtenir les fichiers .ubyte.

🔹 Option 2 : Téléchargement via terminal
Linux / macOS
cd SpikingConvNet-main/data

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz

Windows (PowerShell)
Invoke-WebRequest -Uri http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -OutFile train-images-idx3-ubyte.gz
Invoke-WebRequest -Uri http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -OutFile train-labels-idx1-ubyte.gz
Invoke-WebRequest -Uri http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -OutFile t10k-images-idx3-ubyte.gz
Invoke-WebRequest -Uri http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -OutFile t10k-labels-idx1-ubyte.gz


Ensuite décompressez les fichiers avec 7-Zip
 ou un équivalent.

🔹 Organisation des fichiers

Les 4 fichiers .ubyte doivent être accessibles par les scripts Python.
Nous recommandons de les placer dans le dossier data/ :

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

