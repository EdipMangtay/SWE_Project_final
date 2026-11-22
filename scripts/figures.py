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
    fig, ax = plt.subplots(figsize=(13.6, 5.55), dpi=220)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set(xlim=(0, 13.6), ylim=(0, 5.55))
    ax.axis("off")

    GREEN = "#086654"
    MINT = "#D9F2E4"
    BORDER = "#9AB5A8"
    RUST = "#853C2D"
    ROSE = "#F6E7E1"
    SUB = "#445E57"

    def zone(x, y, w, h, label, bg="#F0F6F2", edge="#C2D6CB"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.10",
                facecolor=bg,
                edgecolor=edge,
                lw=1.1,
                linestyle="--",
            )
        )
        ax.text(
            x + 0.14,
            y + h - 0.19,
            label,
            ha="left",
            va="center",
            fontsize=8.2,
            weight="bold",
            color=GREEN,
        )

    def card(
        x,
        y,
        w,
        h,
        title,
        lines,
        fill="white",
        edge=BORDER,
        lw=1.3,
        title_color=INK,
        badge=None,
        badge_col=GREEN,
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor=fill,
                edgecolor=edge,
                lw=lw,
            )
        )
        ty = y + h - 0.22 if lines else y + h / 2
        ax.text(
            x + 0.13,
            ty,
            title,
            ha="left",
            va="center",
            fontsize=9.0,
            weight="bold",
            color=title_color,
        )
        if badge:
            bw = len(badge) * 0.068 + 0.18
            ax.add_patch(
                FancyBboxPatch(
                    (x + w - bw - 0.10, ty - 0.10),
                    bw,
                    0.20,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    facecolor=badge_col,
                    edgecolor="none",
                )
            )
            ax.text(
                x + w - bw / 2 - 0.10,
                ty,
                badge,
                ha="center",
                va="center",
                fontsize=7.0,
                weight="bold",
                color="white",
            )
        for idx, line in enumerate(lines):
            ax.text(
                x + 0.13,
                ty - 0.21 - idx * 0.175,
                line,
                ha="left",
                va="center",
                fontsize=7.55,
                color=SUB,
            )

    def arrow(p1, p2, col=TEAL, lw=1.45):
        ax.annotate(
            "",
            xy=p2,
            xytext=p1,
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.45,head_width=0.25",
                color=col,
                lw=lw,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    zone(0.08, 0.15, 2.25, 5.25, "1 · CLIENT UI (Browser)", bg="#EEF5F0", edge="#B8CEC2")
    zone(
        2.48,
        0.15,
        2.55,
        5.25,
        "2 · PREPROCESSING (sentinel/text.py)",
        bg="#F2F7F4",
        edge="#B8CEC2",
    )
    zone(
        5.18,
        0.15,
        4.95,
        5.25,
        "3 · MODEL LADDER ON LOCAL CPU (Apple M3 Pro · 4 Threads)",
        bg="#EAF3ED",
        edge="#90B4A4",
    )
    zone(
        10.28,
        0.15,
        3.24,
        5.25,
        "4 · POST-PROCESSING & SESSION",
        bg="#EEF5F0",
        edge="#B8CEC2",
    )

    card(
        0.22,
        3.65,
        1.97,
        1.32,
        "Chat Input & Controls",
        [
            "• st.chat_input (live message)",
            "• Sidebar active model selector",
            "• 'Compare all four models' btn",
            "• 'Clear session' reset button",
        ],
        fill="white",
        edge=TEAL,
        lw=1.4,
    )
    card(
        0.22,
        0.35,
        1.97,
        2.95,
        "Streamlit UI Panels",
        [
            "• Message Review Feed:",
            "  CLEAN (visible) / OFFENSIVE",
            "  ('•••• Message hidden' mask)",
            "• Per-message confidence (%)",
            "  & CPU inference latency (ms)",
            "• Side-by-Side 4-Model Table",
            "  + Edit-distance rule badge",
            "• Sidebar Response Time Log",
            "  & Session Mean Latency",
        ],
        fill="white",
        edge=BORDER,
    )
    card(
        2.62,
        3.92,
        2.27,
        1.05,
        "Input Validation",
        [
            "• validate() in sentinel/text.py",
            "• Non-empty visible string check",
            "• Length cap: 1 to 4,096 chars",
        ],
        fill="white",
        edge=TEAL,
    )
    card(
        2.62,
        2.22,
        2.27,
        1.38,
        "Transformer Branch",
        [
            "• clean_text(): Unicode NFKC",
            "  & whitespace normalization",
            "• Preserves case & punctuation",
            "• BertTokenizer (Cased WordPiece,",
            "  119,547 vocab, max 64 tokens)",
        ],
        fill="#E6F4EC",
        edge=GREEN,
        lw=1.4,
    )
    card(
        2.62,
        0.35,
        2.27,
        1.55,
        "Classical & Neural Branch",
        [
            "• tokens(): NFKC + casefold()",
            "• Turkish folding (İ/ı → i)",
            "• Regex leetspeak repair:",
            "  0→o, 1→i, 3→e, 4→a, 5→s, 7→t",
            "  (e.g., 'id1ot' → 'idiot')",
            "• Repeat squeeze (3+ → 2 chars)",
        ],
        fill="white",
        edge=BORDER,
    )
    card(
        5.34,
        3.92,
        4.63,
        1.05,
        "M4 · Multilingual BERT · INT8",
        [
            "• 74 dynamic INT8 Linear layers (torch.qint8, QNNPACK) + RowInt8Embedding",
            "• Symmetric per-row INT8 word embeddings + FP32 scale · 182.93 MB · 35.04 ms",
        ],
        fill=MINT,
        edge=GREEN,
        lw=1.8,
        badge="DEFAULT · INT8",
        badge_col=GREEN,
    )
    card(
        5.34,
        2.77,
        4.63,
        0.98,
        "M3 · Multilingual BERT · FP32",
        [
            "• bert-base-multilingual-cased (12 layers; 10 frozen, top 2 + pooler + head tuned)",
            "• Full-precision torch.float32 eager attention · 714.36 MB · 25.18 ms",
        ],
        fill="white",
        edge=BORDER,
        badge="FP32",
        badge_col="#54766B",
    )
    card(
        5.34,
        1.62,
        4.63,
        0.98,
        "M1 · TF-IDF + Logistic Regression",
        [
            "• Sublinear TF-IDF word (1,2)-grams (35k vocab, min_df=2) + balanced L2 LogReg",
            "• Double-precision float64 sparse pipeline (C = 4.0) · 0.62 MB · 0.33 ms",
        ],
        fill="white",
        edge=BORDER,
        badge="FP64",
        badge_col="#54766B",
    )
    card(
        5.34,
        0.35,
        4.63,
        1.10,
        "M2 · Own Word2Vec + Feedforward NN",
        [
            "• 64-d Skip-gram (5 neg samples) mean vector → Linear(64,64)-ReLU-Dropout-Linear",
            "• Single-precision float32 network · 2.07 MB · 0.10 ms",
            "• Auxiliary Rule: Levenshtein edit distance ≤ 1 lexicon matcher (comparison panel)",
        ],
        fill="white",
        edge=BORDER,
        badge="FP32 + RULE",
        badge_col="#54766B",
    )
    card(
        10.42,
        3.25,
        2.96,
        1.72,
        "Threshold & CPU Timer",
        [
            "• Offensive probability p ∈ [0, 1]",
            "• Validation-tuned threshold τ:",
            "  BERT FP32/INT8: τ = 0.575",
            "  Word2Vec + NN:  τ = 0.550",
            "  TF-IDF + LogReg: τ = 0.425",
            "• Confidence: p (OFF) or 1−p (CLEAN)",
            "• perf_counter_ns() latency (ms)",
        ],
        fill="white",
        edge=TEAL,
        lw=1.4,
    )
    card(
        10.42,
        1.68,
        2.96,
        1.32,
        "Masking Post-Processor",
        [
            "• displayed(text, offensive):",
            "• If p < τ  → CLEAN:",
            "  Show original validated message",
            "• If p ≥ τ  → OFFENSIVE:",
            "  Replace with '••••  Message hidden'",
        ],
        fill=ROSE,
        edge=RUST,
        lw=1.3,
        title_color=RUST,
    )
    card(
        10.42,
        0.35,
        2.96,
        1.10,
        "In-Memory Session State",
        [
            "• st.session_state.history log",
            "• st.session_state.last_input cache",
            "• Cached CPU adapters (@st.cache_resource)",
            "• Zero disk logging of user chat",
        ],
        fill="white",
        edge=BORDER,
    )

    arrow((2.19, 4.40), (2.62, 4.40))
    arrow((3.75, 3.92), (3.75, 3.60))
    ax.plot([2.54, 2.62], [4.30, 4.30], color=TEAL, lw=1.3)
    ax.plot([2.54, 2.54], [1.12, 4.30], color=TEAL, lw=1.3)
    arrow((2.54, 1.12), (2.62, 1.12))

    ax.plot([4.89, 5.08], [2.95, 2.95], color=GREEN, lw=1.45)
    ax.plot([5.08, 5.08], [3.26, 4.44], color=GREEN, lw=1.45)
    arrow((5.08, 4.44), (5.34, 4.44), col=GREEN)
    arrow((5.08, 3.26), (5.34, 3.26), col=GREEN)

    ax.plot([4.89, 5.08], [1.12, 1.12], color=TEAL, lw=1.45)
    ax.plot([5.08, 5.08], [0.90, 2.11], color=TEAL, lw=1.45)
    arrow((5.08, 2.11), (5.34, 2.11), col=TEAL)
    arrow((5.08, 0.90), (5.34, 0.90), col=TEAL)

    ax.plot([10.12, 10.12], [0.90, 4.44], color=TEAL, lw=1.45)
    for ym in (4.44, 3.26, 2.11, 0.90):
        ax.plot([9.97, 10.12], [ym, ym], color=TEAL, lw=1.45)
    arrow((10.12, 4.10), (10.42, 4.10), col=TEAL)

    arrow((11.90, 3.25), (11.90, 3.00), col=TEAL)
    arrow((11.90, 1.68), (11.90, 1.45), col=TEAL)
    ax.plot([10.42, 10.22], [0.85, 0.85], color=TEAL, lw=1.35)
    ax.plot([10.22, 10.22], [0.06, 0.85], color=TEAL, lw=1.35)
    ax.plot([1.20, 10.22], [0.06, 0.06], color=TEAL, lw=1.35)
    arrow((1.20, 0.06), (1.20, 0.35), col=TEAL)

    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor=PAPER)
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
