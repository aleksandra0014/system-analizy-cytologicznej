import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix_greys(
    cm,
    classes,
    normalize=True,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
    figsize=(6, 5),
    value_fmt=None
):
    """
    Rysuje confusion matrix w stylu jak na przykładzie, w skali szarości.

    Args:
        cm (array-like): confusion matrix (NxN)
        classes (list[str]): etykiety klas (N)
        normalize (bool): normalizacja po wierszach do [0,1]
        title (str): tytuł wykresu
        cmap: colormap (domyślnie Greys)
        figsize (tuple): rozmiar figury
        value_fmt (str|None): format wartości w komórkach,
                              np. ".2f". Jeśli None, dobierany automatycznie.
    """
    cm = np.asarray(cm, dtype=float)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm musi być macierzą kwadratową NxN, dostałem {cm.shape}")
    if len(classes) != cm.shape[0]:
        raise ValueError(f"len(classes) musi równać się N={cm.shape[0]}")

    cm_to_plot = cm.copy()
    if normalize:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        # unikamy dzielenia przez 0
        row_sums[row_sums == 0] = 1.0
        cm_to_plot = cm_to_plot / row_sums

    if value_fmt is None:
        value_fmt = ".2f" if normalize else "d"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        cm_to_plot,
        interpolation="nearest",
        cmap=cmap,
        vmin=0 if normalize else None,
        vmax=1 if normalize else None
    )


    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=270, labelpad=15)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # adnotacje liczbowe w komórkach
    thresh = cm_to_plot.max() / 2.0 if cm_to_plot.size else 0.5
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            val = cm_to_plot[i, j]
            if normalize:
                text = format(val, value_fmt)
                if np.isclose(val, 0):
                    text = "0"
            else:
                text = str(int(round(val)))

            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black"
            )

    ax.set_ylim(len(classes)-0.5, -0.5)  # żeby nie ucinało ramek
    fig.tight_layout()
    plt.show()

    return fig, ax

plot_confusion_matrix_greys(
    cm=np.array([[0.92, 0.0, 0.97],    
                 [0.0, 0.83, 0.03],    
                 [0.08, 0.17, 0.0]]), classes=['cell', 'HSIL_group', 'background'],)