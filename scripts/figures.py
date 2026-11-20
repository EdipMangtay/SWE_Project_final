import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sentinel.config import FINAL, ROOT, read_json

INK = "#182D37"
TEAL = "#087B74"
PALE = "#DDF2EB"
PAPER = "#F8FAF7"


def architecture(out):
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    fig.patch.set_facecolor(PAPER)
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")

    def box(x, y, w, h, title, subtitle="", accent=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.09",
                facecolor=PALE if accent else "white",
                edgecolor=TEAL if accent else "#AEC2C3",
                lw=1.3,
            )
        )
        ax.text(
            x + w / 2,
            y + h * (0.63 if subtitle else 0.5),
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            weight="bold",
            color=INK,
        )
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.29,
                subtitle,
                ha="center",
                va="center",
                fontsize=8.5,
                color=INK,
            )

    def arrow(a, b):
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.4, shrinkA=0, shrinkB=0),
        )

    box(0.1, 2, 1.8, 1, "Chat input", "Browser")
    box(2.25, 2, 1.8, 1, "Validate", "1–4,096 chars")
    arrow((1.94, 2.5), (2.2, 2.5))
    ax.plot([4.07, 4.45], [2.5, 2.5], color=TEAL, lw=1.4)
    ax.plot([4.45, 4.45], [0.75, 4.2], color=TEAL, lw=1.4)
    for y, t, s, h in [
        (3.8, "INT8 multilingual BERT", "Cased WordPiece · default", True),
        (2.65, "FP32 multilingual BERT", "Cased WordPiece", False),
        (1.5, "TF-IDF + logistic regression", "Normalized n-grams · FP64", False),
        (0.35, "Word2Vec + neural net", "Normalized words · FP32", False),
    ]:
        box(4.85, y, 3.2, 0.8, t, s, h)
        arrow((4.45, y + 0.4), (4.8, y + 0.4))
        ax.plot([8.1, 8.4], [y + 0.4, y + 0.4], color=TEAL, lw=1.4)
    ax.plot([8.4, 8.4], [0.75, 4.2], color=TEAL, lw=1.4)
    arrow((8.4, 2.5), (8.8, 2.5))
    box(8.85, 2.0, 2.95, 1, "Threshold and mask", "Class score + CPU time")
    arrow((10.3, 1.95), (10.3, 1.56))
    box(8.85, 0.55, 2.95, 0.95, "Streamlit session", "Chat display + timing log")
    ax.text(
        4.85,
        4.78,
        "Selected model on the local CPU",
        ha="left",
        fontsize=10,
        color=TEAL,
        weight="bold",
    )
    fig.subplots_adjust(0, 0.02, 1, 0.99)
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.05, facecolor=PAPER)
    plt.close(fig)


def generate():
    dest = ROOT / "docs/figures"
    dest.mkdir(exist_ok=True, parents=True)
    architecture(dest / "architecture.png")
    log = read_json(FINAL / "results/neural_training.json")["epochs"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for key, color, label in [
        ("train_loss", TEAL, "Training"),
        ("val_loss", "#AC573B", "Validation"),
    ]:
        ax.plot(
            [r["epoch"] for r in log],
            [r[key] for r in log],
            label=label,
            color=color,
            lw=2,
        )
    ax.set(xlabel="Epoch", ylabel="Weighted binary cross-entropy")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(dest / "training_curve.png", dpi=160)
    plt.close(fig)
    rows = read_json(FINAL / "results/metrics.json")
    fig, axes = plt.subplots(1, 4, figsize=(10.7, 3.0))
    for ax, r, title in zip(
        axes, rows, ["TF-IDF", "Word2Vec + NN", "BERT FP32", "BERT INT8"]
    ):
        a = np.array(r["confusion_matrix"])
        ax.imshow(
            a,
            cmap="GnBu",
            vmin=0,
            vmax=max(max(row) for row in rows[0]["confusion_matrix"]),
        )
        for y, x in np.ndindex(2, 2):
            ax.text(
                x,
                y,
                str(a[y, x]),
                ha="center",
                va="center",
                fontsize=14,
                color="white" if a[y, x] > 200 else INK,
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Clean", "Off."],
            yticklabels=["Clean", "Off."],
            xlabel="Predicted",
            title=title,
        )
    axes[0].set_ylabel("Human label")
    fig.tight_layout()
    fig.savefig(dest / "confusion_matrices.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    generate()
