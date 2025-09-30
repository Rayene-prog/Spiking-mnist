# Import de PyTorch (tenseurs et ops conv/pool utilisés)
import torch
# Import de NumPy pour le calcul vectorisé
import numpy as np
# Barre de progression pour boucles
from tqdm import tqdm
# SVM linéaire pour le readout
from sklearn.svm import LinearSVC
# Métrique d'accuracy
from sklearn.metrics import accuracy_score
# Convolution et max-pooling 2D (ops PyTorch)
from torch.nn.functional import conv2d, max_pool2d

# Chargement + encodage MNIST (spike trains) depuis utils.py
from utils import load_encoded_MNIST


"""
Implementation of the paper STDP-based spiking deep neural networks for object recognition
for the MNIST classification task.

References:

    [1] Kheradpisheh, S. R., Ganjtabesh, M., Thorpe, S. J., & Masquelier, T. (2018).
        STDP-based spiking deep convolutional neural networks for object recognition.
        Neural Networks, 99, 56–67. https://doi.org/10.1016/J.NEUNET.2017.12.005

    [2] Mozafari, M., Ganjtabesh, M., Nowzari-Dalini, A., & Masquelier, T. (2019).
        SpykeTorch: Efficient simulation of convolutional spiking neural networks with
        at most one spike per neuron. Frontiers in Neuroscience, 13, 625.

    [3] https://github.com/npvoid/SDNN_python
"""


# Définition d'une couche de pooling spiking (chaque neurone ne peut tirer qu'une fois)
class SpikingPool:
    """Pooling layer with spiking neurons that can fire only once."""
    # Constructeur : paramétrage des tailles et états
    def __init__(self, input_shape, kernel_size, stride, padding=0):
        # Déballage de la forme d'entrée (C,H,W)
        in_channels, in_height, in_width = input_shape
        # Normalisation des paramètres (scalaires -> tuples)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        # Calcul des dimensions de sortie (formule conv/pool standard)
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width  = int(((in_width  + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        # Mémorisation de la shape de sortie
        self.output_shape = (in_channels, out_height, out_width)
        # Masque des neurones encore autorisés à tirer (un coup max)
        self.active_neurons = np.ones(self.output_shape).astype(bool)  # can fire once

    # Réinitialiser l'état (tous les neurones redeviennent actifs)
    def reset(self):
        self.active_neurons[:] = True

    # Appel : applique un max-pooling spiking en tenant compte des neurones déjà activés
    def __call__(self, in_spks):
        # Padding spatial si demandé
        in_spks = np.pad(in_spks, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant')
        # Conversion en tenseur torch avec axe batch
        in_spks = torch.Tensor(in_spks).unsqueeze(0)
        # Max pooling
        out_spks = max_pool2d(in_spks, self.kernel_size, stride=self.stride).numpy()[0]
        # Ne garder que les spikes des neurones encore actifs
        out_spks = out_spks * self.active_neurons
        # Les neurones qui viennent de tirer deviennent inactifs
        self.active_neurons[out_spks == 1] = False
        # Retourne la carte de spikes pooled
        return out_spks


# Couche convolutionnelle spiking (neurone IF, un spike max) avec apprentissage STDP WTA
class SpikingConv:
    """
    Convolutional layer with IF spiking neurons (one spike max).
    Winner-take-all STDP learning.
    """
    # Constructeur : hyperparamètres réseau + STDP
    def __init__(self, input_shape, out_channels, kernel_size, stride, padding=0,
                 nb_winners=1, firing_threshold=1, stdp_max_iter=None, adaptive_lr=False,
                 stdp_a_plus=0.004, stdp_a_minus=-0.003, stdp_a_max=0.15, inhibition_radius=0,
                 update_lr_cnt=500, weight_init_mean=0.8, weight_init_std=0.05, v_reset=0):
        # Déballer la forme d'entrée
        in_channels, in_height, in_width = input_shape
        # Sorties et hyperparams géométriques
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        # Seuil de tir et reset potentiel
        self.firing_threshold = firing_threshold
        self.v_reset = v_reset
        # Initialisation gaussienne des poids (sera ensuite bornée 0..1 via clip)
        self.weights = np.random.normal(
            loc=weight_init_mean, scale=weight_init_std,
            size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        )

        # Calcul des dimensions spatiales de sortie
        out_height = int(((in_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1)
        out_width  = int(((in_width  + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1)
        # Potentiels membranaires (un par neurone de sortie)
        self.pot = np.zeros((out_channels, out_height, out_width))
        # Masque des neurones actifs (un spike max)
        self.active_neurons = np.ones(self.pot.shape).astype(bool)
        # Shape de sortie (C_out, H_out, W_out)
        self.output_shape = self.pot.shape

        # Mémoire des spikes d'entrée (pour STDP)
        self.recorded_spks = np.zeros((in_channels, in_height + 2*self.padding[0], in_width + 2*self.padding[1]))
        # Paramètres STDP / inhibition
        self.nb_winners = nb_winners
        self.inhibition_radius = inhibition_radius
        self.adaptive_lr = adaptive_lr
        self.a_plus = stdp_a_plus
        self.a_minus = stdp_a_minus
        self.a_max = stdp_a_max
        self.stdp_cnt = 0
        self.update_lr_cnt = update_lr_cnt
        self.stdp_max_iter = stdp_max_iter
        self.plasticity = True
        # Masque des neurones autorisés à faire du STDP
        self.stdp_neurons = np.ones(self.pot.shape).astype(bool)

        # --- Logging (sans impact sur l'apprentissage) ---
        self.log_every = 1
        self.history = {
            "winners": [],          # (t, c, h, w) des neurones gagnants par update
            "num_spikes_t": [],     # (t, n_spikes) à la sortie de cette couche
            "conv_metric": [],      # moyenne de w*(1-w) comme métrique de convergence
            "w_mean": [],           # moyenne des poids
            "w_std":  [],           # écart-type des poids
        }
        # Sauvegarde optionnelle de snapshots de poids
        self.save_weight_snapshots_every = 1000
        self.weight_snapshots = []
        # Compteur de pas de temps
        self.time_step = 0

    # Mesure un indicateur global de "distance à saturation" des poids
    def get_learning_convergence(self):
        return (self.weights * (1 - self.weights)).sum() / np.prod(self.weights.shape)

    # Réinitialise l'état interne de la couche (pour un nouveau sample)
    def reset(self):
        self.pot[:] = self.v_reset
        self.active_neurons[:] = True
        self.stdp_neurons[:] = True
        self.recorded_spks[:] = 0
        self.time_step = 0

    # Sélection des gagnants (WTA) au sein de la carte de potentiels
    def get_winners(self):
        winners = []
        channels = np.arange(self.pot.shape[0])
        # On ignore les neurones STDP interdits via un masque multiplicatif
        pots_tmp = np.copy(self.pot) * self.stdp_neurons
        # Boucle jusqu'à nb_winners ou seuil non atteint
        while len(winners) < self.nb_winners:
            # Neurone ayant le plus grand potentiel
            winner = np.argmax(pots_tmp)
            winner = np.unravel_index(winner, pots_tmp.shape)
            # Si sous le seuil de tir, on s'arrête
            if pots_tmp[winner] <= self.firing_threshold:
                break
            # Ajout du gagnant courant
            winners.append(winner)
            # Inhibition latérale : on remet à v_reset autour du winner (autres canaux)
            pots_tmp[channels != winner[0],
                     max(0, winner[1]-self.inhibition_radius):winner[1]+self.inhibition_radius+1,
                     max(0, winner[2]-self.inhibition_radius):winner[2]+self.inhibition_radius+1] = self.v_reset
            # On reset aussi le canal du winner à cette position
            pots_tmp[winner[0]] = self.v_reset
        # Retour des indices gagnants
        return winners

    # Inhibition latérale "en ligne" à partir d'une carte de spikes binaire
    def lateral_inhibition(self, spks):
        # Indices des spikes
        spks_c, spks_h, spks_w = np.where(spks)
        # Potentiels des positions qui ont spiké
        spks_pot = np.array([self.pot[spks_c[i], spks_h[i], spks_w[i]] for i in range(len(spks_c))])
        # On les trie par potentiel décroissant
        spks_sorted_ind = np.argsort(spks_pot)[::-1]
        # Pour chaque spike (du plus fort au plus faible)
        for ind in spks_sorted_ind:
            if spks[spks_c[ind], spks_h[ind], spks_w[ind]] == 1:
                # Inhibition de la même position (h,w) dans les autres canaux
                inhib_channels = np.arange(spks.shape[0]) != spks_c[ind]
                spks[inhib_channels, spks_h[ind], spks_w[ind]] = 0
                self.pot[inhib_channels, spks_h[ind], spks_w[ind]] = self.v_reset
                self.active_neurons[inhib_channels, spks_h[ind], spks_w[ind]] = False
        # Retourne la carte de spikes après inhibition
        return spks

    # Récupère le patch d'entrée vu par un neurone de sortie (récepteur)
    def get_conv_of(self, input, output_neuron):
        # Déballage : canal + position (h,w) du neurone de sortie
        n_c, n_h, n_w = output_neuron
        # Tenseur 4D (B=1)
        input = torch.Tensor(input).unsqueeze(0)  # batch axis
        # unfold = extraction de tous les patches glissants
        convs = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, stride=self.stride)[0].numpy()
        # Index linéaire du patch correspondant à (n_h, n_w)
        conv_ind = (n_h * self.pot.shape[2]) + n_w
        # Retourne le patch (toutes les entrées de ce champ récepteur)
        return convs[:, conv_ind]

    # Règle de plasticité STDP appliquée au neurone gagnant 'winner'
    def stdp(self, winner):
        # Sécurité : le neurone doit être autorisé au STDP
        if not self.stdp_neurons[winner]:
            exit(1)
        # Si la plasticité est coupée, rien à faire
        if not self.plasticity:
            return
        # Compteur d'updates STDP
        self.stdp_cnt += 1
        # Indices du neurone gagnant
        winner_c, winner_h, winner_w = winner
        # Patch d'entrée associé au winner (spikes enregistrés)
        conv = self.get_conv_of(self.recorded_spks, winner).flatten()
        # Facteur w*(1-w) pour la "sensibilité" au milieu
        w = self.weights[winner_c].flatten() * (1 - self.weights[winner_c]).flatten()
        # Pré/post spikes
        w_plus = conv > 0
        w_minus = conv == 0
        # Mise à jour additive : LTP pour w_plus, LTD pour w_minus
        dW = (w_plus * w * self.a_plus) + (w_minus * w * self.a_minus)
        # Application au filtre du gagnant (reshape)
        self.weights[winner_c] += dW.reshape(self.weights[winner_c].shape)
        # Empêcher les poids de sortir de [0,1]
        np.clip(self.weights[winner_c], 0.0, 1.0, out=self.weights[winner_c])


        # Logging weights + convergence
        try:
            # Métrique de convergence globale
            conv_val = float(self.get_learning_convergence())
            self.history["conv_metric"].append(conv_val)
            # Stats de poids
            W = self.weights
            self.history["w_mean"].append(float(W.mean()))
            self.history["w_std"].append(float(W.std()))
            # Sauvegarde périodique de snapshots
            upd_count = len(self.history["conv_metric"])
            if self.save_weight_snapshots_every and (upd_count % self.save_weight_snapshots_every == 0):
                self.weight_snapshots.append(self.weights[:min(16, self.weights.shape[0])].copy())
        except Exception:
            # Logging robuste : ignorer les erreurs de stats
            pass

        # Interdire le STDP aux neurones dans le voisinage du gagnant (toutes cartes ≠ canal gagnant)
        channels = np.arange(self.pot.shape[0])
        self.stdp_neurons[channels != winner_c,
                          max(0, winner_h-self.inhibition_radius):winner_h+self.inhibition_radius+1,
                          max(0, winner_w-self.inhibition_radius):winner_w+self.inhibition_radius+1] = False
        # Interdire aussi au neurone gagnant lui-même (un gagnant par update)
        self.stdp_neurons[winner_c] = False
        # LR adaptatif : on augmente a_plus (borné), et règle a_minus en conséquence
        if self.adaptive_lr and self.stdp_cnt % self.update_lr_cnt == 0:
            self.a_plus = min(2 * self.a_plus, self.a_max)
            self.a_minus = -0.75 * self.a_plus
        # Éventuel arrêt définitif de la plasticité après N updates
        if self.stdp_max_iter is not None and self.stdp_cnt > self.stdp_max_iter:
            self.plasticity = False

    # Passage avant de la couche sur un timestep (spk_in = cartes de spikes d'entrée)
    def __call__(self, spk_in, train=False):
        # Padding spatial sur l'entrée
        spk_in = np.pad(spk_in, ((0,), (self.padding[0],), (self.padding[1],)), mode='constant')
        # Accumuler les spikes d'entrée (pour STDP)
        self.recorded_spks += spk_in
        # Tenseur de sortie (binaire)
        spk_out = np.zeros(self.pot.shape)
        # Conversion en tenseurs torch
        x = torch.Tensor(spk_in).unsqueeze(0)
        weights = torch.Tensor(self.weights)
        # Convolution 2D (via PyTorch)
        out_conv = conv2d(x, weights, stride=self.stride).numpy()[0]
        # Mise à jour du potentiel uniquement là où les neurones sont encore actifs
        self.pot[self.active_neurons] += out_conv[self.active_neurons]
        # Détection des spikes (seuil)
        output_spikes = self.pot > self.firing_threshold
        # Si au moins un spike
        if np.any(output_spikes):
            # Marquer les spikes de sortie
            spk_out[output_spikes] = 1
            # Appliquer l'inhibition latérale
            spk_out = self.lateral_inhibition(spk_out)
            # Si entraînement + plasticité active, sélectionner les gagnants et appliquer STDP
            if train and self.plasticity:
                winners = self.get_winners()
                # Log des neurones gagnants (avec timestep)
                try:
                    for w in winners:
                        self.history["winners"].append((int(self.time_step), int(w[0]), int(w[1]), int(w[2])))
                except Exception:
                    pass
                # Apprentissage STDP pour chaque gagnant
                for winner in winners:
                    self.stdp(winner)
            # Reset du potentiel des neurones qui ont tiré
            self.pot[spk_out == 1] = self.v_reset
            # Désactiver ces neurones (un seul spike autorisé)
            self.active_neurons[spk_out == 1] = False

        # Logging du nb de spikes à chaque pas de temps
        try:
            if (self.time_step % self.log_every) == 0:
                self.history["num_spikes_t"].append((int(self.time_step), int(spk_out.sum())))
        except Exception:
            pass
        # Incrémenter le compteur de temps
        self.time_step += 1
        # Retourner la carte binaire de spikes de sortie
        return spk_out


# Réseau SNN complet : conv1->pool1->conv2->pool2
class SNN:
    """Spiking convolutional neural network model."""
    # Construction du réseau (initialisation des couches et shapes)
    def __init__(self, input_shape):
        # === conv1 — paper params ===
        conv1 = SpikingConv(
            input_shape,
            out_channels=30, kernel_size=5, stride=1, padding=2,
            nb_winners=1,
            firing_threshold=15,
            stdp_max_iter=None,
            adaptive_lr=True,
            inhibition_radius=2,
            v_reset=0,
            stdp_a_plus=0.004,
            stdp_a_minus=-0.003,
        )

        # Pooling suivant conv1 (2x2, stride 2)
        pool1 = SpikingPool(conv1.output_shape, kernel_size=2, stride=2, padding=0)

        # === conv2 — paper params ===
        conv2 = SpikingConv(
            pool1.output_shape,
            out_channels=100, kernel_size=5, stride=1, padding=2,
            nb_winners=1,
            firing_threshold=10,
            stdp_max_iter=None,
            adaptive_lr=True,
            inhibition_radius=1,
            v_reset=0,
            stdp_a_plus=0.004,
            stdp_a_minus=-0.003,
        )

        # APRES (comme ta version qui marchait bien)
        pool2 = SpikingPool(conv2.output_shape, kernel_size=2, stride=2, padding=0)


        # Enregistrer les listes de couches et la shape finale
        self.conv_layers = [conv1, conv2]
        self.pool_layers = [pool1, pool2]
        self.output_shape = pool2.output_shape
        self.nb_trainable_layers = len(self.conv_layers)
        # Stockage d'un compteur de spikes total par sample (pour stats)
        self.recorded_sum_spks = []

    # Reset de toutes les couches avant un nouveau passage
    def reset(self):
        for layer in self.conv_layers:
            layer.reset()
        for layer in self.pool_layers:
            layer.reset()

    # Passage avant sur une séquence temporelle x (T, C, H, W)
    def __call__(self, x, train_layer=None):
        # Toujours repartir d'un état vierge
        self.reset()
        # Nombre de time-steps
        nb_timesteps = x.shape[0]
        # Pré-allocation sortie temporelle (T, C_out, H_out, W_out)
        output_spikes = np.zeros((nb_timesteps,) + self.output_shape)
        # Compteur de spikes totaux (diagnostic)
        sum_spks = 0
        # Itération temporelle
        for t in range(nb_timesteps):
            # Spikes d'entrée au temps t (float64 pour la conv)
            spk_in = x[t].astype(np.float64)
            sum_spks += spk_in.sum()
            # conv1 (éventuellement en mode apprentissage si train_layer==0)
            spk = self.conv_layers[0](spk_in, train=(train_layer == 0))
            sum_spks += spk.sum()
            # pool1
            spk_in = self.pool_layers[0](spk)
            sum_spks += spk_in.sum()
            # conv2 (éventuellement en mode apprentissage si train_layer==1)
            spk = self.conv_layers[1](spk_in, train=(train_layer == 1))
            sum_spks += spk.sum()
            # pool2 → sortie
            spk_out = self.pool_layers[1](spk)
            sum_spks += spk_out.sum()
            # Stockage du snapshot temporel
            output_spikes[t] = spk_out
        # Si on est en mode inférence (pas d'apprentissage), enregistrer le total de spikes
        if train_layer is None:
            self.recorded_sum_spks.append(sum_spks)
        # Alerte si la sortie n'a aucun spike
        if output_spikes.sum() == 0:
            print("[WARNING] No output spike recorded.")
        # Retour de toute la séquence de sorties
        return output_spikes


# Point d'entrée principal : entraînement STDP couche par couche puis readout SVM
def main(
    seed=1,
    data_prop=1,         # 100% data
    nb_timesteps=30,     # 30 time bins (paper)
    epochs=[2, 2],       # 2 epochs per layer
    convergence_rate=0.01,
):
    # Seeds NumPy/PyTorch pour reproductibilité
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Chargement + encodage du dataset (DoG + encodage temporel)
    X_train, y_train, X_test, y_test = load_encoded_MNIST(data_prop=data_prop, nb_timesteps=nb_timesteps)

    # Initialisation du réseau
    input_shape = X_train[0][0].shape
    snn = SNN(input_shape)

    # Création logs + sauvegarde des poids initiaux (analyse "qui a appris quoi")
    import os
    os.makedirs("logs", exist_ok=True)
    np.save("logs/conv1_weights_init.npy", snn.conv_layers[0].weights.astype(np.float32))
    np.save("logs/conv2_weights_init.npy", snn.conv_layers[1].weights.astype(np.float32))

    # Informations de formes et densité moyenne de spikes en entrée
    print(f"Input shape : {X_train[0].shape} ({np.prod(X_train[0].shape)} values)")
    print(f"Output shape : {snn.output_shape} ({np.prod(snn.output_shape)} values)")
    print(f"Mean spikes count per input : {X_train.mean(0).sum()}")

    ### TRAINING ###
    print("\n### TRAINING ###")
    # Entraînement couche 1 puis couche 2 (STDP non supervisé)
    for layer in range(snn.nb_trainable_layers):
        print(f"Layer {layer+1}...")
        for epoch in range(epochs[layer]):
            print(f"\t epoch {epoch+1}")
            # Parcours des échantillons d'entraînement
            for x, y in zip(tqdm(X_train), y_train):
                # Passe avant et STDP sur la couche 'layer'
                snn(x, train_layer=layer)
                # Arrêt anticipé si la métrique de convergence est passée sous le seuil
                if snn.conv_layers[layer].get_learning_convergence() < convergence_rate:
                    break

    ### TESTING ###
    print("\n### TESTING ###")
    # Extraction des features (MAX et SUM temporels) pour TRAIN
    output_train_max = np.zeros((len(X_train), np.prod(snn.output_shape)))
    output_train_sum = np.zeros((len(X_train), np.prod(snn.output_shape)))
    for i, x in enumerate(tqdm(X_train)):
        spk = snn(x)
        output_train_max[i] = spk.max(0).flatten()
        output_train_sum[i] = spk.sum(0).flatten()

    # Extraction des features (MAX et SUM temporels) pour TEST
    output_test_max = np.zeros((len(X_test), np.prod(snn.output_shape)))
    output_test_sum = np.zeros((len(X_test), np.prod(snn.output_shape)))
    for i, x in enumerate(tqdm(X_test)):
        spk = snn(x)
        output_test_max[i] = spk.max(0).flatten()
        output_test_sum[i] = spk.sum(0).flatten()

    # Stat : nb moyen de spikes totaux par échantillon (utile pour voir l'activité)
    print(f"Mean total number of spikes per sample : {np.mean(snn.recorded_sum_spks)}")

    ### READOUT ###
    # Classifieur linéaire (fidèle au papier : C=2.4)
    clf = LinearSVC(C=2.4, max_iter=3000, random_state=seed)
    clf.fit(output_train_max, y_train)
    y_pred_max = clf.predict(output_test_max)
    acc = accuracy_score(y_test, y_pred_max)
    print(f"Accuracy with method 1 (max) : {acc}")

    # Deuxième lecture : SUM temporel
    clf = LinearSVC(C=2.4, max_iter=3000, random_state=seed)
    clf.fit(output_train_sum, y_train)
    y_pred_sum = clf.predict(output_test_sum)
    acc = accuracy_score(y_test, y_pred_sum)
    print(f"Accuracy with method 2 (sum) : {acc}")

    # --- Logs & weights ---
    # Petite fonction utilitaire pour dumper les historiques d'une couche
    def _dump_layer(tag, layer):
        import numpy as _np
        _np.save(f"logs/{tag}_winners.npy",      _np.array(layer.history['winners'],      dtype=_np.int32))
        _np.save(f"logs/{tag}_num_spikes_t.npy", _np.array(layer.history['num_spikes_t'], dtype=_np.int32))
        _np.save(f"logs/{tag}_conv_metric.npy",  _np.array(layer.history['conv_metric'],  dtype=_np.float32))
        _np.save(f"logs/{tag}_w_mean.npy",       _np.array(layer.history['w_mean'],       dtype=_np.float32))
        _np.save(f"logs/{tag}_w_std.npy",        _np.array(layer.history['w_std'],        dtype=_np.float32))
        # Snapshots de poids (si activés)
        if layer.weight_snapshots:
            _np.save(f"logs/{tag}_weight_snapshots.npy", _np.array(layer.weight_snapshots, dtype=_np.float32))

    # Dump des deux couches
    _dump_layer("conv1", snn.conv_layers[0])
    _dump_layer("conv2", snn.conv_layers[1])

    # Sauvegarde des labels et prédictions (pour matrices de confusion & analyses)
    np.save("logs/y_test.npy",       np.array(y_test, dtype=np.int32))
    np.save("logs/y_pred_max.npy",   np.array(y_pred_max, dtype=np.int32))
    np.save("logs/y_pred_sum.npy",   np.array(y_pred_sum, dtype=np.int32))

    # Sauvegarde des poids appris finaux
    np.save("logs/conv1_weights.npy", snn.conv_layers[0].weights.astype(np.float32))
    np.save("logs/conv2_weights.npy", snn.conv_layers[1].weights.astype(np.float32))

    # Message récap des logs
    print("➡ Logs écrits dans ./logs (activité, convergence, snapshots, poids init/final, y_pred_*)")

    # (Optional) figures si matplotlib est dispo
    try:
        import matplotlib.pyplot as _plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        os.makedirs("figs", exist_ok=True)

        # Courbes de convergence STDP par couche
        for tag in ["conv1", "conv2"]:
            cm = np.load(f"logs/{tag}_conv_metric.npy")
            if cm.size:
                _plt.figure(); _plt.plot(cm)
                _plt.xlabel("updates"); _plt.ylabel("mean w*(1-w)")
                _plt.title(f"{tag.upper()} — convergence STDP"); _plt.tight_layout()
                _plt.savefig(f"figs/{tag}_convergence.png", dpi=140); _plt.close()

        # Matrices de confusion (MAX et SUM)
        y_true = np.load("logs/y_test.npy")
        for tag in ["max", "sum"]:
            yp = np.load(f"logs/y_pred_{tag}.npy")
            cm = confusion_matrix(y_true, yp, labels=list(range(10)))
            disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
            fig, ax = _plt.subplots(figsize=(5,5))
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"Confusion matrix ({tag.upper()})")
            _plt.tight_layout(); _plt.savefig(f"figs/confusion_{tag}.png", dpi=140); _plt.close()

        print("✅ Figures écrites dans ./figs")
    except Exception as _e:
        # Si matplotlib absent ou autre souci de rendu, on n'échoue pas le run
        print(f"[INFO] Figures ignorées ({_e}). Installe matplotlib pour les générer.")


# Lancement direct du script → appelle main() avec hyperparams par défaut
if __name__ == "__main__":
    main()

