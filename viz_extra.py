# viz_extra.py
# Figures propres pour le SNN STDP (MNIST)
# - Convergence STDP par couche
# - Grilles de filtres (moyenne sur canaux d'entrée)
# - Spikes par pas de temps (version simple : bleu clair + moyenne noire)
# - Raster des "winners" (t vs canal)
# - Histogramme des winners par canal (y compris inactives)
# - Exemple d'entrée (somme des spikes ON/OFF sur T) [optionnel]
# - Encodage temporel (frise ON/OFF + courbes)  << AJOUTÉ
# - Activité moyenne par chiffre (10 x C) [silence prints]
# - Matrices de confusion (si y_pred_* présents)

import os
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Réglages globaux ---
mpl.rcParams["agg.path.chunksize"] = 20000  # évite OverflowError sur gros rasters

# --- Dossiers / constantes ---
LOGDIR  = "logs"
FIGDIR  = "figs"
FIG_DPI = 140
NB_TIMESTEPS = 30

# --- Helpers E/S figures ---
os.makedirs(FIGDIR, exist_ok=True)

def _savefig(path):
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")

def _line(x, y, title, xlabel, ylabel, out_png):
    plt.figure()
    plt.plot(x, y, lw=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, linestyle=":")
    _savefig(os.path.join(FIGDIR, out_png))

def _heatmap(mat, title, xlabel, ylabel, out_png, aspect="auto"):
    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect=aspect, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Magnitude")
    _savefig(os.path.join(FIGDIR, out_png))

def _bar(x, h, title, xlabel, ylabel, out_png):
    plt.figure()
    plt.bar(x, h, width=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25, linestyle=":")
    _savefig(os.path.join(FIGDIR, out_png))

def _weights_shape(tag):
    wpath = os.path.join(LOGDIR, f"{tag}_weights.npy")
    if not os.path.exists(wpath):
        return None
    W = np.load(wpath, allow_pickle=True)
    return W.shape

# --- 1) Convergence STDP ---
def plot_convergence(tag):
    path = os.path.join(LOGDIR, f"{tag}_conv_metric.npy")
    if not os.path.exists(path):
        print(f"[WARN] {path} absent."); return
    cm = np.load(path)
    if cm.size == 0:
        print(f"[WARN] {path} vide."); return
    x = np.arange(len(cm))
    _line(x, cm,
          title=f"{tag.upper()} — Convergence STDP (⟨w·(1−w)⟩)",
          xlabel="Nombre de mises à jour (updates)",
          ylabel="⟨w·(1−w)⟩ (adim.)",
          out_png=f"{tag}_convergence_pretty.png")

# --- 2) Grilles de filtres ---
def plot_filters_grids(tag, vmax_mode="per-filter"):
    wpath = os.path.join(LOGDIR, f"{tag}_weights.npy")
    if not os.path.exists(wpath):
        print(f"[WARN] {wpath} absent."); return
    W = np.load(wpath)              # (C, Cin, Kh, Kw)
    C, Cin, Kh, Kw = W.shape
    Wm = W.mean(axis=1)             # (C, Kh, Kw)

    if vmax_mode == "per-filter":
        normed = []
        for i in range(C):
            f = Wm[i]
            lo, hi = f.min(), f.max()
            nf = (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)
            normed.append(nf)
        Wshow = np.stack(normed, axis=0)
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(Wm.min()), float(Wm.max())
        Wshow = Wm

    ncols = int(math.ceil(math.sqrt(C)))
    nrows = int(math.ceil(C / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8*ncols, 1.8*nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    ims = []
    for i in range(nrows*ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.axis("off")
        if i < C:
            im = ax.imshow(Wshow[i], vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"ch {i}", fontsize=8); ims.append(im)
    if vmax_mode != "per-filter" and ims:
        fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.85)
    _savefig(os.path.join(FIGDIR, f"{tag}_filters_grid_pretty.png"))

# --- 3) Spikes par pas de temps (VERSION SIMPLE) ---
def plot_spikes_per_timestep(tag):
    """
    1 figure par couche (tag='conv1' ou 'conv2').
    - SANS labels: toutes les courbes en BLEU CLAIR (même couleur), moyenne noire ± 1 écart-type.
    - AVEC labels (logs/<tag>_num_spikes_labels.npy): couleur = CLASSE + légende des classes.
    Pas de colorbar, pas d'index, pas de limitation de séries.
    """
    path = os.path.join(LOGDIR, f"{tag}_num_spikes_t.npy")
    if not os.path.exists(path):
        print(f"[WARN] {path} absent."); return
    arr = np.load(path)  # attendu: (N, 2) -> (t, n_spikes)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.size == 0:
        print(f"[WARN] format inattendu: {getattr(arr,'shape',None)} — skip."); return

    # Découpe en séries (rupture quand t redescend 29 -> 0)
    t_all = arr[:, 0]
    breaks = np.where(np.diff(t_all) < 0)[0] + 1
    if breaks.size > 0:
        series = np.split(arr, breaks)              # liste de (T_i, 2)
    elif NB_TIMESTEPS > 0 and (arr.shape[0] % NB_TIMESTEPS == 0):
        series = list(arr.reshape(-1, NB_TIMESTEPS, 2))
    else:
        # fallback (tracé simple)
        t, n = arr[:, 0], arr[:, 1]
        _line(t, n,
              title=f"{tag.upper()} — Nombre de spikes par pas",
              xlabel="Temps (pas discrets)",
              ylabel="Nombre de spikes (#)",
              out_png=f"{tag}_spikes_per_timestep_pretty.png")
        return

    K = len(series)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    # Couleurs : labels optionnels
    labels = None
    lab_path = os.path.join(LOGDIR, f"{tag}_num_spikes_labels.npy")
    if os.path.exists(lab_path):
        lab = np.load(lab_path)
        if lab.shape[0] == K:
            labels = lab
        else:
            print(f"[WARN] labels ignorés (taille {lab.shape[0]} != {K}).")

    handles = []
    if labels is None:
        blue = mpl.colormaps.get_cmap("tab10")(0)
        alpha_line = 0.18 if K > 2000 else 0.3
        lw_line = 0.9 if K > 2000 else 1.0
        for seg in series:
            ax.plot(seg[:, 0], seg[:, 1], lw=lw_line, alpha=alpha_line, color=blue)
    else:
        classes = np.unique(labels)
        cmap = (mpl.colormaps.get_cmap("tab10").resampled(len(classes))
                if len(classes) <= 10 else
                mpl.colormaps.get_cmap("tab20").resampled(len(classes)))
        class_to_color = {int(c): cmap(i) for i, c in enumerate(classes)}
        for i, seg in enumerate(series):
            ax.plot(seg[:, 0], seg[:, 1], lw=1.0, alpha=0.55,
                    color=class_to_color[int(labels[i])])
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=class_to_color[int(c)], lw=2,
                          label=f"classe {int(c)}")
                   for c in classes]

    # Moyenne ± écart-type (noir)
    T = min(len(s) for s in series)
    stack = np.vstack([s[:T, 1] for s in series])   # (K, T)
    mean_y, std_y = stack.mean(axis=0), stack.std(axis=0)
    mean_x = series[0][:T, 0]
    mean_line, = ax.plot(mean_x, mean_y, color="k", lw=3, label=f"moyenne ({K} séries)")
    ax.fill_between(mean_x, mean_y - std_y, mean_y + std_y, color="k", alpha=0.12)

    # Légende
    if labels is None:
        ax.legend([mean_line], ["moyenne"], loc="upper right", framealpha=0.9)
    else:
        handles.append(mean_line)
        ax.legend(handles=handles, loc="upper right", framealpha=0.9, title="Code couleur")

    # Axes / titres
    ax.set_xlim(mean_x.min(), mean_x.max())
    ax.set_title(f"{tag.upper()} — Nombre de spikes par pas")
    ax.set_xlabel("Temps (pas discrets)")
    ax.set_ylabel("Nombre de spikes (#)")
    ax.grid(alpha=0.25, linestyle=":")
    _savefig(os.path.join(FIGDIR, f"{tag}_spikes_per_timestep_pretty.png"))

# --- 4) Raster + histogramme des winners ---
def plot_raster_and_counts(tag):
    wpath = os.path.join(LOGDIR, f"{tag}_winners.npy")
    if not os.path.exists(wpath):
        print(f"[WARN] {wpath} absent."); return
    win = np.load(wpath)  # (N, 4) : (t, c, h, w)
    if win.size == 0:
        print(f"[WARN] {wpath} vide."); return
    t = win[:, 0].astype(np.int64)
    c = win[:, 1].astype(np.int64)

    # Raster (downsample si immense)
    N = len(t)
    stride = max(1, N // 800_000)
    plt.figure(figsize=(8, 4))
    plt.scatter(t[::stride], c[::stride], s=2, alpha=0.35, linewidths=0)
    plt.title(f"{tag.upper()} — Raster des winners")
    plt.xlabel("Temps (pas)")
    plt.ylabel("Canal gagnant (#)")
    plt.grid(alpha=0.25, linestyle=":")
    _savefig(os.path.join(FIGDIR, f"{tag}_raster_pretty.png"))

    # Histogramme des canaux (y compris inactifs)
    wshape = _weights_shape(tag)
    nch = wshape[0] if wshape is not None else int(c.max()) + 1
    counts = np.bincount(c, minlength=nch)
    x = np.arange(nch)
    _bar(x, counts,
         title=f"{tag.upper()} — Nombre de victoires par canal",
         xlabel="Canal (#)",
         ylabel="Victoires (#)",
         out_png=f"{tag}_channel_counts_pretty.png")

# --- 5) Exemple d'entrée (optionnel) ---
def plot_input_spikes_example():
    try:
        from utils import load_encoded_MNIST
    except Exception as e:
        print(f"[WARN] Exemple d'entrée ignoré (MNIST manquant) : {e}")
        return
    X_tr, y_tr, X_te, y_te = load_encoded_MNIST(data_prop=1.0, nb_timesteps=NB_TIMESTEPS)
    x = X_te[0]  # (T, 2, 28, 28)
    sum_on  = x[:, 0].sum(axis=0)
    sum_off = x[:, 1].sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), constrained_layout=True)
    im0 = axes[0].imshow(sum_on,  interpolation="nearest")
    axes[0].set_title("Somme spikes canal ON")
    axes[0].set_xlabel("x (pixel)"); axes[0].set_ylabel("y (pixel)")
    c0 = fig.colorbar(im0, ax=axes[0], shrink=0.8); c0.set_label("Spikes (#)")

    im1 = axes[1].imshow(sum_off, interpolation="nearest")
    axes[1].set_title("Somme spikes canal OFF")
    axes[1].set_xlabel("x (pixel)"); axes[1].set_ylabel("y (pixel)")
    c1 = fig.colorbar(im1, ax=axes[1], shrink=0.8); c1.set_label("Spikes (#)")
    _savefig(os.path.join(FIGDIR, "input_spikes_example_pretty.png"))

# --- 5bis) Encodage temporel (frise ON/OFF + courbes) ---  << AJOUTÉ
def plot_input_encoding_timeline(n_steps=8, pick="quantiles"):
    """
    Montre l'encodage temporel d'UN exemple :
    - Rangée 1 : cartes ON à différents temps t
    - Rangée 2 : cartes OFF aux mêmes temps
    - Bas : courbes #spikes ON(t) et OFF(t)
    n_steps: nb d'instantanés affichés
    pick: "first" -> t=0..; "quantiles" -> instants répartis sur [0..T-1]
    """
    try:
        from utils import load_encoded_MNIST
    except Exception as e:
        print(f"[WARN] Timeline ignorée (MNIST manquant) : {e}")
        return

    # Charge un exemple encodé
    X_tr, y_tr, X_te, y_te = load_encoded_MNIST(data_prop=1.0, nb_timesteps=NB_TIMESTEPS)
    x = X_te[0]                     # (T, 2, 28, 28)
    T = x.shape[0]
    on  = x[:, 0]                   # (T, H, W)
    off = x[:, 1]

    # Temps à afficher
    if pick == "first":
        t_idx = np.arange(min(n_steps, T))
    else:  # "quantiles"
        t_idx = np.unique(np.linspace(0, T-1, n_steps, dtype=int))

    # Bornes communes par rangée (échelle stable)
    vmin_on,  vmax_on  = float(on.min()),  float(on.max())
    vmin_off, vmax_off = float(off.min()), float(off.max())

    # Layout : 2 rangées d'images + 1 axe pour les courbes
    fig = plt.figure(figsize=(1.8*len(t_idx), 6))
    gs  = mpl.gridspec.GridSpec(3, len(t_idx), height_ratios=[1, 1, 0.9], hspace=0.25)

    axes_on  = [fig.add_subplot(gs[0, i]) for i in range(len(t_idx))]
    axes_off = [fig.add_subplot(gs[1, i]) for i in range(len(t_idx))]
    ax_curve = fig.add_subplot(gs[2, :])

    ims_on, ims_off = [], []
    for i, t in enumerate(t_idx):
        im0 = axes_on[i].imshow(on[t],  vmin=vmin_on,  vmax=vmax_on,  interpolation="nearest")
        axes_on[i].set_title(f"t={int(t)}", fontsize=9)
        axes_on[i].set_xticks([]); axes_on[i].set_yticks([])
        ims_on.append(im0)

        im1 = axes_off[i].imshow(off[t], vmin=vmin_off, vmax=vmax_off, interpolation="nearest")
        axes_off[i].set_xticks([]); axes_off[i].set_yticks([])
        ims_off.append(im1)

    # Labels de rangée (gauche)
    axes_on[0].set_ylabel("ON",  rotation=0, labelpad=15, va="center", fontsize=10)
    axes_off[0].set_ylabel("OFF", rotation=0, labelpad=12, va="center", fontsize=10)

    # Colorbars (une par rangée)
    fig.colorbar(ims_on[0],  ax=axes_on,  shrink=0.8).set_label("Spikes (#)")
    fig.colorbar(ims_off[0], ax=axes_off, shrink=0.8).set_label("Spikes (#)")

    # Courbes globales ON/OFF vs temps
    s_on  = on.sum(axis=(1, 2))
    s_off = off.sum(axis=(1, 2))
    tt = np.arange(T)
    ax_curve.plot(tt, s_on,  lw=2, label="ON")
    ax_curve.plot(tt, s_off, lw=2, label="OFF")
    for t in t_idx:
        ax_curve.axvline(int(t), color="k", alpha=0.12, lw=1)
    ax_curve.set_xlabel("Temps (pas discrets)")
    ax_curve.set_ylabel("Spikes par pas (#)")
    ax_curve.grid(alpha=0.25, linestyle=":")
    ax_curve.legend()

    fig.suptitle("Encodage temporel — exemple (ON/OFF)", y=0.98)
    _savefig(os.path.join(FIGDIR, "input_encoding_timeline_pretty.png"))

# --- 6) Activité moyenne par chiffre (silence prints) ---
def plot_activity_per_digit(n_max=300):
    """Carte (10 x C) de l'activité moyenne par chiffre.
       Silence les messages verbeux pendant l'inférence.
       n_max: limite d'exemples test (None => tous)."""
    try:
        import torch
        from snn import SNN
        from utils import load_encoded_MNIST
    except Exception as e:
        print(f"[WARN] Activité par chiffre ignorée (dépendance manquante) : {e}")
        return

    X_tr, y_tr, X_te, y_te = load_encoded_MNIST(data_prop=1.0, nb_timesteps=NB_TIMESTEPS)
    if n_max is not None:
        X_te = X_te[:n_max]; y_te = y_te[:n_max]

    snn = SNN(X_tr[0][0].shape)
    w1 = os.path.join(LOGDIR, "conv1_weights.npy")
    w2 = os.path.join(LOGDIR, "conv2_weights.npy")
    if not (os.path.exists(w1) and os.path.exists(w2)):
        print("[WARN] Poids finaux absents — lance d'abord `python snn.py`."); return
    snn.conv_layers[0].weights = np.load(w1)
    snn.conv_layers[1].weights = np.load(w2)
    for c in snn.conv_layers:
        c.plasticity = False

    C = snn.output_shape[0]
    mat = np.zeros((10, C), dtype=float)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        for x, y in zip(X_te, y_te):
            spk = snn(x)                         # (T, C, H, W) ou (T, C, 1, 1)
            v   = spk.max(0).max(axis=(-2, -1))  # (C,)
            mat[y] += v

    counts = np.bincount(y_te, minlength=10).reshape(-1, 1).astype(float)
    counts[counts == 0] = 1.0
    mat = mat / counts

    _heatmap(mat,
             title="Activité moyenne par chiffre (max_t puis max_(x,y))",
             xlabel="Carte (#)",
             ylabel="Chiffre (0–9)",
             out_png="activity_per_digit_heatmap_pretty.png")

# --- 7) Matrices de confusion ---
def plot_confusions_pretty():
    y_true_p = os.path.join(LOGDIR, "y_test.npy")
    if not os.path.exists(y_true_p):
        print(f"[WARN] {y_true_p} absent (lance d’abord `python snn.py`)."); return
    y_true = np.load(y_true_p)

    for tag in ["max", "sum"]:
        yp_p = os.path.join(LOGDIR, f"y_pred_{tag}.npy")
        if not os.path.exists(yp_p):
            print(f"[WARN] {yp_p} absent."); continue
        yp = np.load(yp_p)
        cm = confusion_matrix(y_true, yp, labels=list(range(10)))
        acc = (yp == y_true).mean()

        disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Matrice de confusion — LinearSVC ({tag.upper()})\nAccuracy = {acc:.4f}")
        _savefig(os.path.join(FIGDIR, f"confusion_{tag}_pretty.png"))

# --- MAIN ---
def main():
    plot_convergence("conv1"); plot_convergence("conv2")
    plot_filters_grids("conv1", vmax_mode="per-filter")
    plot_filters_grids("conv2", vmax_mode="per-filter")
    plot_spikes_per_timestep("conv1"); plot_spikes_per_timestep("conv2")
    plot_raster_and_counts("conv1");  plot_raster_and_counts("conv2")
    plot_input_spikes_example()
    plot_input_encoding_timeline(n_steps=8, pick="quantiles")   # << APPEL AJOUTÉ
    plot_activity_per_digit()
    plot_confusions_pretty()
    print("✅ Terminé.")

if __name__ == "__main__":
    main()

