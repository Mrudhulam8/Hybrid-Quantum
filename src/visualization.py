"""Publication-quality figures (Phase 3: confusion matrix; others in Phase 15)."""


# ---------------------------------------------------------------------------
# Phase 3: confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names, path, title="Confusion matrix", dev_watermark=False):
    """Row-normalised heat-map annotated with raw counts. Returns the saved path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(3.4, 0.9 * n + 1.5),) * 2)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    if dev_watermark:
        fig.text(0.5, 0.5, "NOT PAPER-ELIGIBLE", rotation=30, ha="center", va="center",
                 fontsize=18, color="red", alpha=0.25)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300); plt.close(fig)
    return path
