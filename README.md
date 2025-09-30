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


## Encodage en spikes (automatique)

Les images MNIST ne sont pas utilisées directement.
Elles sont encodées en trains de spikes ON/OFF sur T=30 pas de temps (par défaut).

Exemple :

from utils import load_encoded_MNIST
X_tr, y_tr, X_te, y_te = load_encoded_MNIST(data_prop=1.0, nb_timesteps=30)
# X_* : (N, T, 2, 28, 28), y_* : labels 0–9

##  Entraînement (STDP)

Lancer l’apprentissage :

python snn.py


Apprentissage STDP : uniquement sur X_tr, y_tr (60k images train, ou moins si réduit avec data_prop)

Évaluation linéaire (LinearSVC) : appliquée sur le test set (X_te, y_te) avec deux descripteurs :

MAX : maximum temporel

SUM : somme temporelle

Sorties principales (dans logs/) :

conv1_weights.npy, conv2_weights.npy : poids appris

conv{1,2}_conv_metric.npy : métrique de convergence (⟨w·(1−w)⟩)

y_test.npy : labels vrais du test set

y_pred_max.npy, y_pred_sum.npy : prédictions SVM linéaire

## Évaluation complémentaire
1) Readout linéaire (inclus dans snn.py)

Les prédictions y_pred_max.npy et y_pred_sum.npy sont produites automatiquement pendant l’entraînement.

2) Readout SVM RBF (optionnel)

Pour tester un classifieur non-linéaire :

python eval_rbf.py


Génère figs/confusion_rbf_max.png et figs/confusion_rbf_sum.png

Sauvegarde les features dans logs/features_rbf.npz

## Visualisations

Centralisées dans viz_extra.py :

python viz_extra.py


Figures générées (dans figs/) :

Convergence STDP : métrique ⟨w·(1−w)⟩

Grilles de filtres appris

Nombre de spikes par pas de temps

Raster des winners + histogramme des canaux

Encodage temporel d’un exemple

Activité moyenne par chiffre

Matrices de confusion (linéaire MAX/SUM)

### Contrôle de la taille d’entraînement

Paramètre data_prop dans load_encoded_MNIST (dans snn.py) :

1.0 → utilise 100% du train set (60k images)

0.1 → utilise 10% (≈6000 images)

0.01 → utilise 1% (≈600 images)

Utile pour tester rapidement ou faire des expériences contrôlées.

### Organisation des sorties

logs/

poids, métriques, spikes, prédictions

figs/

PNG haute résolution pour toutes les analyses

### Reproductibilité : ordre conseillé
#### 1) Installer les dépendances
pip install -r requirements.txt

#### 2) Télécharger MNIST
#### (voir section "Données d’entrée")

#### 3) Entraîner le réseau
python snn.py

#### 4) Générer les figures
python viz_extra.py

#### 5) (optionnel) Évaluer avec SVM RBF
python eval_rbf.py

### Résultats typiques

Les performances varient selon les hyperparamètres et la taille de l’échantillon :

Readout linéaire (MAX/SUM) : ~85–90% d’accuracy sur le test set (10k)

Readout RBF (optionnel) : légèrement supérieur (≈90–92%)

Ces valeurs sont indicatives : utilisez viz_extra.py pour analyser vos propres runs.

### Références

MNIST — Yann LeCun et al. : http://yann.lecun.com/exdb/mnist/

SNN + STDP :

Bi & Poo (1998), Synaptic modifications in cultured hippocampal neurons

Diehl & Cook (2015), Unsupervised learning of digit recognition using STDP in spiking neural networks



