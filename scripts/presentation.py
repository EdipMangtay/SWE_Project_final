"""Regenerate the 10-slide PDF after updating docs/submission.json."""

from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import simpleSplit
from sentinel.config import FINAL, ROOT, read_json
from scripts.figures import generate

W, H = 960, 540
INK = "#152D39"
TEAL = "#087B74"
MUTED = "#46636A"
PAPER = "#F8FAF7"
LINE = "#C9D7D3"
ACCENT = "#BBEBDD"


def build():
    results = FINAL / "results"
    rows = read_json(results / "metrics.json")
    by = {r["model"]: r for r in rows}
    data = read_json(results / "dataset.json")
    bench = read_json(results / "benchmark.json")
    neural_epochs = read_json(results / "neural_training.json")["epochs"]
    bert_epochs = read_json(results / "bert_training.json")["epochs"]
    neural_best = max(neural_epochs, key=lambda row: row["val_macro_f1"])["epoch"]
    bert_best = max(bert_epochs, key=lambda row: row["val_macro_f1"])["epoch"]
    cover = read_json(ROOT / "docs/submission.json")
    c = Canvas(str(ROOT / "docs/presentation.pdf"), pagesize=(W, H), pageCompression=1)
    c.setTitle("SentinelAI | Real-Time Chat Moderation")
    c.setAuthor(cover["name"])
    fig = ROOT / "docs/figures"
    if not (fig / "confusion_matrices.png").exists():
        generate()

    def fill(color):
        c.setFillColor(colors.HexColor(color))

    def txt(x, y, s, sz=18, bold=False, col=INK):
        fill(col)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", sz)
        c.drawString(x, H - y, s)

    def line(x, y, xx, yy, color=LINE, width=1):
        c.setStrokeColor(colors.HexColor(color))
        c.setLineWidth(width)
        c.line(x, H - y, xx, H - yy)

    def paragraph(x, y, s, width, sz=18, leading=24, col=INK, bold=False):
        for row in simpleSplit(s, "Helvetica-Bold" if bold else "Helvetica", sz, width):
            txt(x, y, row, sz, bold, col)
            y += leading
        return y

    def start(no, part, title):
        fill(PAPER)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        txt(47, 46, part, 14, True, TEAL)
        txt(47, 100, title, 35, True)
        line(47, 116, 913, 116)
        line(47, 488, 913, 488)
        txt(47, 513, "SWE015   /   SentinelAI", 12, False, MUTED)
        txt(884, 513, f"{no:02d}", 12, False, MUTED)

    def table(y, heads, colx, values, width=866, step=44, fs=17, highlight=-1):
        for x, h in zip(colx, heads):
            txt(x, y, h, 13, True, TEAL)
        line(47, y + 13, 47 + width, y + 13, TEAL)
        for i, row in enumerate(values):
            yy = y + step * (i + 1)
            if i == highlight:
                fill("#DDF2EB")
                c.rect(46, H - yy - 12, width, step - 2, fill=1, stroke=0)
            for x, s in zip(colx, row):
                txt(x, yy, str(s), fs, i == highlight)
            line(47, yy + 14, 47 + width, yy + 14)

    def img(path, x, y, w, h):
        c.drawImage(
            str(path),
            x,
            H - y - h,
            width=w,
            height=h,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )

    def linked(x, y, label, url):
        txt(x, y, label, 16, True, ACCENT)
        if url:
            c.linkURL(
                url,
                (
                    x,
                    H - y - 4,
                    x + c.stringWidth(label, "Helvetica-Bold", 16),
                    H - y + 17,
                ),
            )

    fp, q = by["bert"], by["int8"]
    reduction = 100 * (1 - q["size_mb"] / fp["size_mb"])
    fill(INK)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    txt(55, 68, "SWE015  /  Introduction to Large Language Models", 17, True, ACCENT)
    txt(51, 176, "SentinelAI", 79, True, "#FFFFFF")
    txt(56, 231, "Real-Time Intelligent Chat Moderation", 31, False, "#FFFFFF")
    txt(57, 281, "Use Case 4   /   English community messages", 20, False, ACCENT)
    line(55, 330, 902, 330, "#59747A")
    txt(56, 375, cover["name"], 25, True, "#FFFFFF")
    txt(
        56,
        409,
        f"Student ID   {cover['student_id'] or 'Pending'}",
        17,
        False,
        "#DDECE8",
    )
    linked(
        56,
        470,
        "Open code repository"
        if cover["repository_url"]
        else "Repository link pending",
        cover["repository_url"],
    )
    linked(
        497,
        470,
        "Watch the two-minute demo" if cover["video_url"] else "Video link pending",
        cover["video_url"],
    )
    c.showPage()
    start(2, "Introduction", "What happens when a player sends a message?")
    txt(49, 172, "01", 24, True, TEAL)
    txt(110, 172, "Read the message", 24, True)
    paragraph(
        110,
        204,
        "A player types in English. The selected CPU model estimates whether the message is offensive.",
        770,
    )
    txt(49, 280, "02", 24, True, TEAL)
    txt(110, 280, "Decide what players see", 24, True)
    paragraph(
        110,
        312,
        "Clean text appears in the chat. Flagged text is masked. The sidebar records the duration of every decision.",
        770,
    )
    txt(49, 390, "03", 24, True, TEAL)
    txt(110, 390, "Measure success", 24, True)
    paragraph(
        110,
        422,
        "Use macro-F1 for the two classes. Set a mean CPU budget of 50 ms per message before benchmarking.",
        770,
    )
    c.showPage()
    start(3, "Development", "From six labels to two classes")
    paragraph(
        48,
        157,
        f"Jigsaw supplied {data['raw']:,} labeled Wikipedia comments. Cleaning and conflict removal left {data['eligible']:,}. We sampled {data['sample']:,} to fit the local training budget.",
        855,
        18,
        25,
    )
    values = []
    for k, label in [
        ("train", "Training"),
        ("validation", "Validation"),
        ("test", "Held-out test"),
    ]:
        s = data["splits"][k]
        values.append(
            [
                label,
                f"{s['rows']:,}",
                f"{s['clean']:,}",
                s["offensive"],
                f"{s['offensive_pct']:.1f}%",
            ]
        )
    table(
        241,
        ["Split", "Messages", "Clean", "Offensive", "Offensive share"],
        [52, 285, 455, 598, 747],
        values,
        step=43,
    )
    paragraph(
        48,
        419,
        "A positive value in any of the six toxicity labels becomes OFFENSIVE. Stratification preserved the class balance. Seed 42 fixed the same test IDs for all models.",
        860,
        17,
        23,
    )
    c.showPage()
    start(4, "Development", "The path from a message to a decision")
    img(fig / "architecture.png", 47, 129, 866, 293)
    paragraph(
        48,
        451,
        "The browser displays results. Preprocessing, inference and masking run in the local Python process. Comparison sends the last input through each branch.",
        855,
        16,
        21,
    )
    c.showPage()
    start(5, "Development", "Technique map")
    vals = [
        (
            "A",
            "Neural networks",
            "Train skip-gram Word2Vec, average the vectors and update a feedforward classifier by backpropagation.",
        ),
        (
            "B",
            "Classical NLP",
            "Normalize noisy spellings. Compare a one-edit lexicon rule with learned word n-gram TF-IDF and logistic regression.",
        ),
        (
            "C",
            "Transformers",
            "Use cased WordPiece. Fine-tune two multilingual BERT encoder blocks, the pooler and a binary classification head.",
        ),
        (
            "D",
            "Efficient models",
            "Quantize the linear layers and word embeddings. Reload the artifact and benchmark single-message inference on CPU.",
        ),
    ]
    for i, (letter, name, detail) in enumerate(vals):
        yy = 162 + 82 * i
        txt(50, yy, letter, 27, True, TEAL)
        txt(100, yy, name, 20, True)
        paragraph(330, yy, detail, 578, 17, 22)
        line(48, yy + 43, 912, yy + 43)
    c.showPage()
    start(6, "Development", "How the models learned")
    txt(48, 166, "TF-IDF + logistic regression", 21, True)
    paragraph(
        48,
        198,
        "Fit unigrams and bigrams on training comments. Select C and the decision threshold on validation macro-F1.",
        426,
        16,
        21,
    )
    txt(48, 286, "Word2Vec + neural network", 21, True)
    paragraph(
        48,
        317,
        f"Train 64-dimensional embeddings on the same messages. AdamW fits a 64-unit ReLU hidden layer. Validation selected epoch {neural_best}.",
        426,
        16,
        21,
    )
    txt(48, 402, "Multilingual BERT", 21, True)
    paragraph(
        48,
        431,
        f"Freeze the lower ten blocks. Fine-tune for three epochs at 4e-5, using at most 64 tokens. Validation selected epoch {bert_best}.",
        426,
        16,
        21,
    )
    img(fig / "training_curve.png", 492, 172, 416, 224)
    txt(510, 417, "Neural network learning curve", 15, True, TEAL)
    c.showPage()
    start(7, "Development", "What changed when BERT was quantized?")
    txt(49, 180, f"{reduction:.1f}%", 57, True, TEAL)
    txt(50, 215, "less space on disk", 21, True)
    paragraph(
        51,
        257,
        "PyTorch dynamically quantizes 74 linear layers. The word embedding uses an INT8 code and a scale for each vocabulary row.",
        387,
        18,
        24,
    )
    paragraph(
        51,
        376,
        "Layer normalization and other remaining operations use floating point.",
        387,
        17,
        22,
        MUTED,
    )
    for i, (title, model) in enumerate([("FP32", fp), ("INT8", q)]):
        yy = 184 + i * 103
        txt(500, yy, title, 19, True)
        txt(735, yy, f"{model['size_mb']:.1f} MB", 19, True)
        fill("#BDD8D1")
        c.rect(500, H - yy - 31, 365, 12, fill=1, stroke=0)
        fill(TEAL)
        c.rect(
            500,
            H - yy - 31,
            365 * model["size_mb"] / fp["size_mb"],
            12,
            fill=1,
            stroke=0,
        )
    paragraph(
        500,
        390,
        f"The INT8 mean was {q['mean_ms']:.2f} ms, versus {fp['mean_ms']:.2f} ms for FP32 on the same CPU.",
        383,
        18,
        24,
    )
    c.showPage()
    start(8, "Development", "Four models, the same 500 test messages")
    names = [
        "TF-IDF + logistic regression",
        "Word2Vec + neural network",
        "Multilingual BERT FP32",
        "Multilingual BERT INT8",
    ]
    table(
        170,
        ["Model", "Macro-F1", "Accuracy", "Size (MB)", "Mean (ms)"],
        [53, 396, 537, 674, 806],
        [
            [
                name,
                f"{r['macro_f1']:.3f}",
                f"{r['accuracy']:.1%}",
                f"{r['size_mb']:.2f}",
                f"{r['mean_ms']:.2f}",
            ]
            for name, r in zip(names, rows)
        ],
        step=48,
        highlight=3,
        fs=17,
    )
    paragraph(
        50,
        412,
        f"{bench['hardware']['cpu']}, four PyTorch threads, batch size one. Each model timed the same 150 held-out messages after 15 warm-ups. Sizes include preprocessing, tokenizer and metadata where required.",
        860,
        17,
        23,
    )
    c.showPage()
    start(9, "Conclusion", "What the moderation system missed")
    txt(50, 170, f"{q['fn']} missed offensive messages", 27, True)
    paragraph(
        50,
        205,
        f"INT8 identified {q['tp']} of the 50 offensive comments. It also masked {q['fp']} of the 450 clean comments.",
        850,
        19,
        26,
    )
    example = read_json(results / "failure_example.json")
    line(50, 254, 911, 254)
    txt(51, 291, "An actual false negative", 16, True, TEAL)
    paragraph(50, 333, "“" + example["excerpt"] + "”", 840, 25, 31, bold=True)
    txt(
        50,
        378,
        f"Human label: OFFENSIVE     Model: CLEAN     Offensive score: {example['offensive_score']:.1%}",
        17,
        False,
        MUTED,
    )
    paragraph(
        50,
        428,
        "The training data comes from Wikipedia discussions. Before applying this model to games, test short chat messages, spelling changes and conversation context.",
        845,
        17,
        23,
    )
    c.showPage()
    start(10, "Conclusion", "AI usage and sources")
    txt(50, 170, "How AI assisted", 22, True)
    paragraph(
        50,
        205,
        "OpenAI ChatGPT / Codex assisted with implementation, debugging, documentation and slide drafting. Training logs, test predictions, file sizes and CPU timings come from executing this project.",
        850,
        18,
        25,
    )
    txt(50, 304, "Sources", 22, True)
    for i, (label, url) in enumerate(
        [
            (
                "Jigsaw / Conversation AI — Toxic Comment Classification Challenge",
                "https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge",
            ),
            (
                "Heliosoph — mirror of Kaggle's labeled training file and data attribution",
                "https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments",
            ),
            (
                "Google BERT — pretrained multilingual cased checkpoint",
                "https://huggingface.co/google-bert/bert-base-multilingual-cased",
            ),
            (
                "PyTorch — dynamic quantization documentation",
                "https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html",
            ),
        ]
    ):
        yy = 344 + 32 * i
        txt(50, yy, label, 16, False, TEAL)
        c.linkURL(
            url,
            (50, H - yy - 2, 50 + c.stringWidth(label, "Helvetica", 16), H - yy + 15),
        )
    c.showPage()
    c.save()


if __name__ == "__main__":
    build()
