# eval_rbf.py
# Évaluation d’un SNN entraîné (poids figés) avec un classifieur SVM RBF
# Génère matrices de confusion pour deux types de features (MAX et SUM)
# Sauvegarde également les features extraites dans logs/features_rbf.npz

import os, numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# --- Constantes ---
NB_TIMESTEPS = 30     # nombre de pas temporels utilisés lors de l’encodage
SPLIT_PROP   = 1.0    # proportion de données à charger (1.0 = toutes)
SEED         = 1      # graine aléatoire pour reproductibilité

# --- Imports locaux ---
from snn import SNN                    # modèle spiking convolutionnel
from utils import load_encoded_MNIST   # fonction d’encodage MNIST en spikes

def load_trained_snn():
    """
    Charge le SNN avec ses poids entraînés (fichiers logs/conv{1,2}_weights.npy).
    Les couches sont figées (plasticité désactivée).
    Retourne : snn, X_tr, y_tr, X_te, y_te
    """
    X_tr, y_tr, X_te, y_te = load_encoded_MNIST(data_prop=SPLIT_PROP, nb_timesteps=NB_TIMESTEPS)
    snn = SNN(X_tr[0][0].shape)  # instancie le modèle selon la shape d’entrée
    
    # chemins des poids appris
    w1, w2 = "logs/conv1_weights.npy", "logs/conv2_weights.npy"
    if not (os.path.exists(w1) and os.path.exists(w2)):
        raise FileNotFoundError("Poids introuvables. Lance d'abord `python snn.py`.")
    
    # charge les poids dans les couches convolutives
    snn.conv_layers[0].weights = np.load(w1)
    snn.conv_layers[1].weights = np.load(w2)
    
    # désactive la plasticité (pas d’apprentissage supplémentaire)
    for c in snn.conv_layers: 
        c.plasticity = False
    
    return snn, X_tr, y_tr, X_te, y_te

def features_from_snn(snn, X):
    """
    Extrait deux types de features pour chaque entrée :
    - Fmax : activation maximale dans le temps pour chaque carte
    - Fsum : somme temporelle des activations
    Retourne (Fmax, Fsum) de taille (N, C*H*W)
    """
    Fmax = np.zeros((len(X), np.prod(snn.output_shape)))
    Fsum = np.zeros((len(X), np.prod(snn.output_shape)))
    for i, x in enumerate(X):
        spk = snn(x)                # sortie temporelle (T, C, H, W)
        Fmax[i] = spk.max(0).flatten()
        Fsum[i] = spk.sum(0).flatten()
    return Fmax, Fsum

def main():
    # --- Chargement modèle et données ---
    snn, X_tr, y_tr, X_te, y_te = load_trained_snn()
    
    # extraction des features sur train/test
    Ftr_max, Ftr_sum = features_from_snn(snn, X_tr)
    Fte_max, Fte_sum = features_from_snn(snn, X_te)

    # --- SVM RBF sur features MAX ---
    clf = SVC(kernel="rbf", gamma="scale", C=10, random_state=SEED)
    clf.fit(Ftr_max, y_tr)
    yp = clf.predict(Fte_max)
    acc = accuracy_score(y_te, yp)
    print(f"Accuracy RBF (MAX): {acc:.4f}")
    
    # matrice de confusion (MAX)
    cm = confusion_matrix(y_te, yp, labels=list(range(10)))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(values_format='d', cmap="Blues", colorbar=False)
    os.makedirs("figs", exist_ok=True)
    plt.title("Confusion matrix — SVM RBF (MAX)")
    plt.tight_layout()
    plt.savefig("figs/confusion_rbf_max.png", dpi=140)
    plt.close()

    # --- SVM RBF sur features SUM ---
    clf.fit(Ftr_sum, y_tr)
    yp = clf.predict(Fte_sum)
    acc = accuracy_score(y_te, yp)
    print(f"Accuracy RBF (SUM): {acc:.4f}")
    
    # matrice de confusion (SUM)
    cm = confusion_matrix(y_te, yp, labels=list(range(10)))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(values_format='d', cmap="Blues", colorbar=False)
    plt.title("Confusion matrix — SVM RBF (SUM)")
    plt.tight_layout()
    plt.savefig("figs/confusion_rbf_sum.png", dpi=140)
    plt.close()

    # --- Sauvegarde des features extraites ---
    np.savez_compressed("logs/features_rbf.npz",
                        Ftr_max=Ftr_max, Ftr_sum=Ftr_sum, y_tr=y_tr,
                        Fte_max=Fte_max, Fte_sum=Fte_sum, y_te=y_te)

if __name__ == "__main__":
    main()

