import numpy as np                          # Import du module NumPy pour les opérations numériques
from scipy.ndimage import correlate          # Import de la corrélation 2D (convolution) depuis SciPy
from math import ceil                        # Import de la fonction ceil (plafond) depuis math
from mnist import MNIST                      # Import du loader MNIST du package python-mnist


def spike_encoding(img, nb_timesteps):
    """
    Encode an image into spikes using a temporal coding based on pixel intensity.
    
    Args : 
        img (ndarray) : input of shape (height,width)
        nb_timesteps (int) : number of spike bins
    """
    # Encodage intensité→latence : plus l’intensité est grande, plus le spike survient tôt.
    with np.errstate(divide='ignore',invalid='ignore'): # Désactive les warnings de division par zéro / invalid
        I, lat = np.argsort(1/img.flatten()), np.sort(1/img.flatten())  # Trie des pixels par 1/intensité (latence)
    # Suppression des pixels de valeur 0 (latence = inf quand 1/0)
    I = np.delete(I, np.where(lat == np.inf))
    # Conversion des indices 1D (dans le flatten) vers des coordonnées 2D (ligne, colonne)
    II = np.unravel_index(I, img.shape)
    # Calcul du pas temporel pour chaque pixel trié afin de répartir sur nb_timesteps-1
    t_step = np.ceil(np.arange(I.size) / (I.size / (nb_timesteps-1))).astype(np.uint8)
    # Ajout de la dimension temps devant les indices spatiaux (forme attendue par l’indexation)
    # shape finale des indices : (timestep, height, width)
    II = (t_step,) + II
    # Création du tenseur binaire de spikes (T × H × W)
    spike_times = np.zeros((nb_timesteps, img.shape[0], img.shape[1]), dtype=np.uint8)
    spike_times[II] = 1  # Place un spike (=1) aux positions (t, y, x) calculées
    return spike_times    # Retourne le cube temporel de spikes (binaire)



def DoG_filter(img, filt, threshold):
    """
    Apply a DoG filter on the given image. 
    
    Args : 
        img (ndarray) : input of shape (height,width)
        filt (ndarray) : DoG filter
        threshold (int) : threshold applied on contrasts
    """
    # Applique la corrélation 2D (équivalente à une convolution retournée) avec padding constant
    img = correlate(img, filt, mode='constant')
    # Met les bords à 0 pour éviter les artefacts (garde un cadre intérieur de 5 pixels)
    border = np.zeros(img.shape)
    border[5:-5, 5:-5] = 1.
    img = img * border
    # Seuil : ne conserve que les valeurs au-dessus du threshold (sinon 0)
    img = (img >= threshold).astype(int) * img
    # Valeur absolue (contrastes positifs)
    img = np.abs(img)
    return img  # Image filtrée et seuillée



def DoG(size, s1, s2):
    """
    Create a DoG filter. 
    
    Args : 
        size (int) : size of the filter
        s1 (int) : std1
        s2 (int) : std2
    """
    # Crée des grilles x,y de 1..size
    r = np.arange(size)+1
    x = np.tile(r, [size, 1])
    y = x.T
    # Distance au centre au carré (coordonnées centrées à moitié pixel près)
    d2 = (x-size/2.-0.5)**2 + (y-size/2.-0.5)**2
    # Différence de Gaussiennes normalisée (1/√(2π)) avec écarts s1 et s2
    filt = 1/np.sqrt(2*np.pi) * (1/s1 * np.exp(-d2/(2*(s1**2))) - 1/s2 * np.exp(-d2/(2*(s2**2))))
    # Centrage (moyenne nulle)
    filt -= np.mean(filt[:])
    # Normalisation par le max (amplitude dans [-1,1])
    filt /= np.amax(filt[:])
    return filt  # Filtre DoG (2D) prêt à l’emploi



def preprocess_MNIST(dataset, nb_timesteps, filters, threshold):
    """
    Preprocess the MNIST dataset. 
    """
    # Nombre de canaux = nombre de filtres DoG utilisés (ex : 2)
    nb_channels = len(filters)
    # Récupère (N, H, W)
    samples, height, width = dataset.shape
    # Pré-alloue la sortie : (N, T, C, H, W) en uint8 (binaire)
    out = np.zeros((samples, nb_timesteps, nb_channels, height, width), dtype=np.uint8)
    # Boucle sur chaque image brute
    for i,img in enumerate(dataset):
        # Tampon par image : (C, T, H, W) en float par défaut
        encoded_img = np.zeros((nb_channels, nb_timesteps, height, width))
        # Applique chaque filtre DoG et encode en spikes temporels
        for f,filt in enumerate(filters):
            dog_img = DoG_filter(img, filt, threshold)           # Contrastes via DoG + seuil
            encoded_img[f] = spike_encoding(dog_img, nb_timesteps)  # Encodage temporel binaire
        # Réordonne (C, T, H, W) -> (T, C, H, W) et stocke
        out[i] = np.swapaxes(encoded_img,0,1)
    return out  # Jeu encodé pour tout le dataset



def load_MNIST(data_prop=1):
    """
    Load the MNIST dataset. 
    """
    # Initialise le loader MNIST (cherche les fichiers ubyte dans le répertoire courant)
    mndata = MNIST()
    images, labels = mndata.load_training()  # Charge train (images, labels) au format listes
    
    # Ensemble d’entraînement
    X_train, y_train = np.asarray(images), np.asarray(labels)  # Conversion en ndarrays
    if data_prop < 1:
        # Échantillonne une proportion data_prop du train sans remise
        samples_ind = np.random.choice(len(X_train), int(len(X_train)*data_prop), replace=False)
        X_train = X_train[samples_ind]
        y_train = y_train[samples_ind]
    # Reshape des vecteurs 784 en images 28×28
    X_train = X_train.reshape(-1, 28, 28)
    # Mélange aléatoire (permutation des indices)
    random_indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[random_indices], y_train[random_indices]

    # Ensemble de test
    images, labels = mndata.load_testing()   # Charge test
    X_test, y_test = np.asarray(images), np.asarray(labels)
    if data_prop < 1:
        # Échantillonnage proportionnel sur le test également
        samples_ind = np.random.choice(len(X_test), int(len(X_test)*data_prop), replace=False)
        X_test = X_test[samples_ind]
        y_test = y_test[samples_ind]
    # Reshape en 28×28
    X_test = X_test.reshape(-1, 28, 28)

    # Récupère la forme d’entrée (28,28) pour information
    input_shape = X_test[0].shape

    return X_train, y_train, X_test, y_test, input_shape  # Retourne jeux + shape



def load_encoded_MNIST(data_prop=1, nb_timesteps=15, threshold=15, filters=[DoG(7,1,2),DoG(7,2,1)]):
    """
    Load and preprocess the MNIST dataset. 
    """
    # Charge MNIST brut (train/test) selon la proportion demandée
    X_train, y_train, X_test, y_test, _ = load_MNIST(data_prop)
    # Encode le train avec filtres DoG + encodage temporel
    X_train_encoded = preprocess_MNIST(X_train, nb_timesteps, filters, threshold)
    # Encode le test de la même manière (mêmes hyperparamètres)
    X_test_encoded = preprocess_MNIST(X_test, nb_timesteps, filters, threshold)
    # Retourne les tenseurs encodés + labels
    return X_train_encoded, y_train, X_test_encoded, y_test

