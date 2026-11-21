"""Build the assignment PDF. Use --powerpoint with python-pptx to also export editable slides."""

import argparse
from shutil import copyfile

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from sentinel.config import FINAL, ROOT, read_json

W, H = 960, 540
PAPER, WHITE = "#F8FAF7", "#FFFFFF"
INK, FOREST, GREEN = "#182D2B", "#103E36", "#086654"
MINT, FAINT, RULE = "#BCEDD1", "#E5F1E8", "#C4D5C9"
SUB, LIGHT = "#47615A", "#D3E6DA"
RUST, ROSE = "#853C2D", "#F5E8DF"
SERIF, SANS, BOLD, ITALIC = "Times-Roman", "Helvetica", "Helvetica-Bold", "Times-Italic"
PPT_FONTS = {
    SERIF: "Times New Roman",
    ITALIC: "Times New Roman",
    SANS: "Arial",
    BOLD: "Arial",
}
STEM = "SentinelAI_Ali_Edip_Mangtay_210911044"


class Slides:
    """Draw from the same point coordinates in PDF and native PowerPoint shapes."""

    def __init__(self, cover, powerpoint):
        self.pdf_path = ROOT / "docs/presentation.pdf"
        self.c = Canvas(str(self.pdf_path), pagesize=(W, H), pageCompression=1)
        self.c.setAuthor(cover["name"])
        self.c.setTitle("SentinelAI | Real-time chat moderation")
        self.c.setSubject(
            "SWE015 • Use Case 4 • Held-out evaluation and CPU deployment"
        )
        self.ppt = None
        if powerpoint:
            from pptx import Presentation
            from pptx.util import Pt

            self.ppt = Presentation()
            self.ppt.slide_width, self.ppt.slide_height = Pt(W), Pt(H)
            self.ppt.core_properties.author = cover["name"]
            self.ppt.core_properties.title = "SentinelAI | Real-time chat moderation"
            self.ppt.core_properties.subject = "SWE015 | Use Case 4"

    def new(self, bg=PAPER):
        if self.ppt is not None:
            from pptx.dml.color import RGBColor

            self.slide = self.ppt.slides.add_slide(self.ppt.slide_layouts[6])
            self.slide.background.fill.solid()
            self.slide.background.fill.fore_color.rgb = RGBColor.from_string(bg[1:])
        self.c.setFillColor(HexColor(bg))
        self.c.rect(0, 0, W, H, stroke=0, fill=1)

    def box(self, x, y, w, h, color):
        self.c.setFillColor(HexColor(color))
        self.c.rect(x, H - y - h, w, h, stroke=0, fill=1)
        if self.ppt is not None:
            from pptx.dml.color import RGBColor
            from pptx.enum.shapes import MSO_SHAPE
            from pptx.util import Pt

            s = self.slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, *map(Pt, (x, y, w, h)))
            s.fill.solid()
            s.fill.fore_color.rgb = RGBColor.from_string(color[1:])
            s.line.fill.background()
            s.shadow.inherit = False

    def rule(self, x, y, w, color=RULE, weight=1):
        self.box(x, y, w, weight, color)

    def text(
        self, x, y, value, size=18, font=SANS, color=INK, *, right=False, link=None
    ):
        # y is the baseline. Keep each line editable and hyperlinks on the actual text.
        val = str(value)
        width = stringWidth(val, font, size)
        if right:
            x -= width
        if x < -0.1 or x + width > W + 0.1 or y > H:
            raise ValueError(f"Text outside slide at ({x}, {y}): {val}")
        self.c.setFillColor(HexColor(color))
        self.c.setFont(font, size)
        self.c.drawString(x, H - y, val)
        if link:
            self.c.linkURL(
                link, (x, H - y - 3, x + width, H - y + size), relative=0, thickness=0
            )
        if self.ppt is not None:
            from pptx.dml.color import RGBColor
            from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
            from pptx.util import Pt

            w = min(W - x, max(12, width * 1.08 + 5))
            shape = self.slide.shapes.add_textbox(
                *map(Pt, (x, y - size * 0.90, w, size * 1.4))
            )
            shape.name = val
            tf = shape.text_frame
            tf.word_wrap = False
            tf.auto_size = MSO_AUTO_SIZE.NONE
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Pt(0)
            tf.vertical_anchor = MSO_ANCHOR.TOP
            p = tf.paragraphs[0]
            p.space_before = p.space_after = Pt(0)
            run = p.add_run()
            run.text = val
            f = run.font
            f.name, f.size = PPT_FONTS[font], Pt(size)
            f.bold, f.italic = font == BOLD, font == ITALIC
            if link:
                run.hyperlink.address = link
            f.color.rgb = RGBColor.from_string(color[1:])

    def para(self, x, y, value, w, size=18, leading=25, font=SANS, color=INK):
        for line in simpleSplit(value, font, size, w):
            self.text(x, y, line, size, font, color)
            y += leading
        return y

    def image(self, path, x, y, w):
        iw, ih = ImageReader(str(path)).getSize()
        h = w * ih / iw
        self.c.drawImage(str(path), x, H - y - h, w, h)
        if self.ppt is not None:
            from pptx.util import Pt

            self.slide.shapes.add_picture(str(path), *map(Pt, (x, y, w, h)))

    def heading(self, n, section, topic, title):
        self.new()
        self.text(46, 38, section.upper(), 12, BOLD, GREEN)
        self.text(914, 38, topic, 13, SANS, SUB, right=True)
        self.rule(46, 52, 868)
        self.text(44, 109, title, 43, SERIF)
        self.rule(46, 495, 868)
        self.text(46, 520, "SentinelAI  /  SWE015", 11.5, SANS, SUB)
        self.text(914, 520, f"{n:02d} / 10", 11.5, SANS, SUB, right=True)

    def end(self, notes):
        if self.ppt is not None:
            self.slide.notes_slide.notes_text_frame.text = notes
        self.c.showPage()

    def save(self):
        self.c.save()
        delivery = ROOT / "delivery"
        delivery.mkdir(exist_ok=True)
        copyfile(self.pdf_path, delivery / f"{STEM}.pdf")
        if self.ppt is not None:
            self.ppt.save(delivery / f"{STEM}.pptx")


def build(powerpoint=False):
    res = FINAL / "results"
    models = {r["model"]: r for r in read_json(res / "metrics.json")}
    tf, nn, fp, q = [models[k] for k in ("tfidf", "neural", "bert", "int8")]
    bench = read_json(res / "benchmark.json")
    data = read_json(res / "dataset.json")
    tst = data["splits"]["test"]
    cover = read_json(ROOT / "docs/submission.json")
    neural_epochs = read_json(res / "neural_training.json")["epochs"]
    neural_best = max(neural_epochs, key=lambda e: e["val_macro_f1"])
    bert = read_json(res / "bert_training.json")
    bert_best = max(bert["epochs"], key=lambda e: e["val_macro_f1"])
    baseline = max(
        read_json(res / "tfidf_training.json"), key=lambda e: e["validation_macro_f1"]
    )
    quant = read_json(res / "int8_reload.json")
    rule = read_json(res / "rule_baseline.json")
    bad = read_json(res / "failure_example.json")
    demos = {r["input"]: r["prediction"] for r in read_json(res / "demo_examples.json")}
    cpu, budget = bench["hardware"]["cpu"], bench["budget_ms"]
    reduction = 100 * (1 - q["size_mb"] / fp["size_mb"])
    repo = cover["repository_url"]
    s = Slides(cover, powerpoint)

    # Cover: the small diagram is the same moderation action shown in the live demo.
    s.new(FOREST)
    s.text(48, 52, "SWE015   /   INTRODUCTION TO LARGE LANGUAGE MODELS", 13, BOLD, MINT)
    s.rule(48, 75, 864, "#497467")
    s.text(44, 197, "SentinelAI", 91, SERIF, WHITE)
    s.text(49, 250, "Real-time chat moderation", 32, SANS, WHITE)
    s.para(
        50,
        296,
        "Can a model flag an offensive game chat message within 50 milliseconds on a CPU?",
        449,
        20,
        28,
        color=LIGHT,
    )
    s.box(555, 119, 355, 239, "#E3F0E4")
    s.text(580, 157, "RECORDED INT8 PREDICTION", 12, BOLD, GREEN)
    s.text(578, 207, "“You are a stupid id1ot.”", 25, ITALIC, FOREST)
    s.rule(580, 227, 304, "#B6CDBB")
    s.box(580, 249, 14, 14, FOREST)
    s.box(601, 249, 14, 14, FOREST)
    s.box(622, 249, 14, 14, FOREST)
    s.box(643, 249, 14, 14, FOREST)
    s.text(580, 299, "Message hidden", 24, BOLD, FOREST)
    s.text(
        580,
        328,
        f"OFFENSIVE    /    {demos['You are a stupid id1ot.']['confidence']:.1%} confidence",
        14,
        SANS,
        GREEN,
    )
    s.rule(48, 395, 864, "#497467")
    s.text(49, 434, cover["name"], 23, BOLD, WHITE)
    s.text(
        49, 464, f"Student ID {cover['student_id']}    /    Use Case 4", 16, SANS, LIGHT
    )
    s.text(555, 424, "GitHub repository", 16, BOLD, WHITE, link=repo)
    s.text(555, 443, repo, 12.5, SANS, MINT, link=repo)
    if cover["video_url"]:
        s.text(
            555,
            474,
            "Watch the two-minute demo",
            16,
            BOLD,
            WHITE,
            link=cover["video_url"],
        )
        s.text(
            555,
            493,
            cover["video_url"],
            12.5,
            SANS,
            MINT,
            link=cover["video_url"],
        )
    else:
        s.text(555, 474, "Demo video link pending", 16, SANS, MINT)
    s.end(
        "I built SentinelAI for a gaming chat moderator. My question was whether a smaller BERT could flag offensive English messages within a mean CPU budget of 50 milliseconds."
    )

    s.heading(2, "Introduction", "Application", "What should other players see?")
    s.para(
        46,
        153,
        "Players type in English. The app checks each new message before displaying it in the chat.",
        840,
        20,
        26,
    )
    for x, raw, tag, col, tint in [
        (46, "Thanks for your help!", "CLEAN", GREEN, FAINT),
        (491, "You are a stupid id1ot.", "OFFENSIVE", RUST, ROSE),
    ]:
        pred = demos[raw]
        s.box(x, 192, 423, 197, tint)
        s.text(x + 20, 225, "PLAYER TYPES", 12, BOLD, col)
        s.text(x + 18, 265, "“" + raw + "”", 28, ITALIC)
        s.rule(x + 20, 285, 380, "#B7C8BD" if tag == "CLEAN" else "#D6C0B2")
        s.text(
            x + 20,
            317,
            f"{tag}   /   {pred['confidence']:.2%} confidence",
            16,
            BOLD,
            col,
        )
        result = pred["display"].replace("••••  ", "")
        s.text(x + 20, 358, "Chat shows", 15, SANS, SUB)
        s.text(x + 110, 358, result, 20, BOLD)
    s.text(46, 431, "Success criteria", 20, BOLD)
    s.para(
        224,
        431,
        f"Detect offensive messages while keeping clean messages visible. Measure both classes with macro-F1 and aim for a mean under {budget} ms per message on CPU.",
        686,
        18,
        25,
    )
    s.end(
        "These are two recorded predictions from the default INT8 model. In the video I type both messages live. For each one the sidebar records the model and elapsed time. I can also switch models or compare them on the same input."
    )

    s.heading(
        3,
        "Development",
        "Data and evaluation",
        "Only one in ten test messages was offensive",
    )
    s.para(
        46,
        150,
        f"Jigsaw supplied {data['raw']:,} labeled English Wikipedia comments. I removed duplicates and conflicting labels, then sampled {data['sample']:,} of the {data['eligible']:,} eligible comments.",
        865,
        18,
        24,
    )
    s.text(46, 222, "Stratified split", 21, BOLD)
    s.text(558, 222, f"Seed {data['seed']}", 14, SANS, SUB, right=True)
    cols = [51, 332, 438, 555]
    for i, lab in enumerate(("Purpose", "Messages", "Clean", "Offensive")):
        s.text(cols[i], 256, lab, 14, BOLD, SUB, right=i > 0)
    s.rule(46, 268, 512)
    for i, (key, name) in enumerate(
        [
            ("train", "Train models"),
            ("validation", "Tune and select"),
            ("test", "Final test"),
        ]
    ):
        row = data["splits"][key]
        y = 301 + i * 42
        if key == "test":
            s.box(46, y - 26, 512, 38, FAINT)
        for j, item in enumerate(
            [name, f"{row['rows']:,}", f"{row['clean']:,}", f"{row['offensive']:,}"]
        ):
            s.text(
                cols[j], y, item, 18, BOLD if key == "test" else SANS, INK, right=j > 0
            )
    s.box(594, 210, 320, 189, FOREST)
    s.text(614, 245, "WHY MACRO-F1?", 13, BOLD, MINT)
    s.text(612, 299, f"{tst['clean'] / tst['rows']:.0%}", 52, SERIF, WHITE)
    s.para(
        614,
        331,
        "accuracy is possible by calling every message clean. Macro-F1 gives each class equal weight.",
        277,
        17,
        23,
        color=WHITE,
    )
    s.para(
        46,
        442,
        "Any of the six toxicity flags makes a comment offensive. All six must be zero for a clean label. The four models saw the same held-out test IDs.",
        860,
        17,
        24,
        color=SUB,
    )
    s.end(
        "The data comes from the Jigsaw Toxic Comment Classification Challenge. To avoid leakage I grouped normalized duplicates before splitting. Training fits model parameters, validation selects thresholds and checkpoints, and the test set measures the final comparison. A clean-only model would miss all fifty offensive examples."
    )

    s.heading(
        4,
        "Development",
        "System architecture",
        "From a new message to a moderation decision",
    )
    s.image(ROOT / "docs/figures/architecture.png", 58, 124, 844)
    s.para(
        46,
        456,
        "The browser shows the label, confidence and time for each message. Model selection switches the CPU branch. Compare runs the last input through all four models and shows the separate lexicon check.",
        870,
        16,
        22,
        color=SUB,
    )
    s.end(
        "Read from left to right. The app validates the input. TF-IDF and Word2Vec normalize noisy words. BERT preserves case and tokenizes with WordPiece. A model-specific threshold selects the label. The app masks flagged text and logs the latency. The independent lexicon comparison never overrides the selected model."
    )

    s.heading(5, "Development", "Required techniques", "Technique map")
    s.text(46, 153, "Area", 14, BOLD, SUB)
    s.text(297, 153, "Where the technique appears in SentinelAI", 14, BOLD, SUB)
    s.rule(46, 168, 868, GREEN, 1.4)
    techniques = [
        (
            "A",
            "Neural networks",
            "Skip-gram Word2Vec learns word vectors from training messages. A feedforward network classifies their mean vector.",
        ),
        (
            "B",
            "Classical NLP",
            "Regex handles noisy spellings. A one-edit lexicon checks for insults. A separate TF-IDF classifier learns word and phrase weights.",
        ),
        (
            "C",
            "Transformers",
            "Cased WordPiece tokenization feeds multilingual BERT. Its final two encoder blocks and classification head learn the task.",
        ),
        (
            "D",
            "Efficient models",
            "INT8 weights shrink BERT. The app reloads the quantized model and runs it locally on the CPU.",
        ),
    ]
    for i, (letter, area, desc) in enumerate(techniques):
        y = 204 + i * 74
        s.text(46, y + 4, letter, 29, SERIF, GREEN)
        s.text(90, y, area, 19, BOLD)
        s.para(297, y - 1, desc, 610, 17.5, 22)
        s.rule(46, y + 39, 868)
    s.end(
        f"The edit-distance rule is a separate reference point. On its own it caught {rule['tp']} of the {tst['offensive']} offensive comments. The four-model comparison uses TF-IDF as its classical baseline. Training, fine-tuning and quantization are separate steps."
    )

    s.heading(6, "Development", "Training protocol", "How the classifiers learned")
    entries = [
        (
            "Classical baseline",
            "TF-IDF + logistic regression",
            f"Fit word unigrams and bigrams on training text. Validation selected C = {baseline['C']:g} and a decision threshold of {baseline['threshold']:.3f}.",
        ),
        (
            "Neural network",
            "Word2Vec + feedforward net",
            f"Train 64-dimensional word vectors from scratch. Average them per message, then train a 64-unit ReLU network with dropout and backpropagation. Keep epoch {neural_best['epoch']}.",
        ),
        (
            "Transformer",
            "Multilingual BERT",
            f"Freeze the lower {bert['frozen_lower_blocks']} blocks. Fine-tune the top two, pooler and classifier using AdamW and up to {bert['tokens']} tokens. Validation selected epoch {bert_best['epoch']}.",
        ),
    ]
    for i, (family, name, desc) in enumerate(entries):
        y = 153 + 102 * i
        s.text(46, y + 3, family.upper(), 12, BOLD, GREEN)
        s.text(46, y + 37, name, 22, SERIF)
        s.para(397, y + 5, desc, 510, 17.5, 23)
        s.rule(46, y + 74, 868)
    s.box(46, 441, 868, 38, FAINT)
    s.text(57, 466, "Fourth model", 15, BOLD, GREEN)
    s.text(
        164,
        466,
        "Convert the selected BERT checkpoint to INT8 and keep its validation threshold.",
        16,
        SANS,
        INK,
    )
    s.end(
        "I used the same train/validation split for the three fitted classifiers. TF-IDF counts word patterns. Word2Vec learns its own word vectors from the training set. Multilingual BERT starts with pretrained weights. The fourth model is a conversion of the selected BERT checkpoint, so it inherits the same decision threshold."
    )

    s.heading(
        7, "Development", "Quantization", "A smaller BERT met the CPU time budget"
    )
    s.box(46, 148, 355, 268, FOREST)
    s.text(67, 185, "MODEL STORAGE", 13, BOLD, MINT)
    s.text(63, 263, f"{reduction:.1f}%", 73, SERIF, WHITE)
    s.text(68, 299, "less disk space", 24, BOLD, WHITE)
    s.para(
        68,
        342,
        f"I quantized {quant['linear_modules']} linear layers and stored word embeddings as INT8 with a scale per row.",
        303,
        17,
        22,
        color=LIGHT,
    )

    def bars(y, label, field, scale, unit):
        s.text(440, y, label, 19, BOLD)
        for i, (name, row, col) in enumerate(
            [("FP32", fp, "#69867C"), ("INT8", q, GREEN)]
        ):
            yy = y + 34 + i * 36
            s.text(440, yy, name, 15, BOLD, SUB)
            s.box(498, yy - 11, 274, 10, FAINT)
            s.box(498, yy - 11, 274 * row[field] / scale, 10, col)
            s.text(
                908, yy, f"{row[field]:.2f} {unit}", 18, BOLD if i else SANS, right=True
            )

    bars(176, "Complete model size", "size_mb", fp["size_mb"], "MB")
    bars(315, f"Mean time per message  /  {budget} ms budget", "mean_ms", budget, "ms")
    s.para(
        46,
        454,
        f"On the {cpu}, INT8 took {q['mean_ms'] - fp['mean_ms']:.2f} ms longer per message on average. Its 95th-percentile time was {q['p95_ms']:.2f} ms.",
        855,
        17,
        23,
        color=SUB,
    )
    s.end(
        "Quantization stores many weights with fewer bits. Layer normalization and remaining operations still use floating point. The saved model shrank by about three quarters. Full-precision BERT was faster on this CPU. The INT8 mean met the target, and individual messages can take longer."
    )

    s.heading(
        8, "Development", "Held-out results", "Four models on the same 500 messages"
    )
    s.text(
        46,
        149,
        f"{cpu}    /    {bench['hardware']['torch_threads']} CPU threads    /    {bench['runs']} timed inputs per model after {bench['warmup']} warm-ups    /    batch size {bench['batch_size']}",
        16,
        SANS,
        SUB,
    )
    xs = [55, 493, 627, 765, 902]
    for i, h in enumerate(["Model", "Macro-F1", "Accuracy", "Size (MB)", "Mean (ms)"]):
        s.text(xs[i], 198, h, 16, BOLD, GREEN, right=i > 0)
    s.rule(46, 211, 868, GREEN, 1.5)
    labels = [
        "TF-IDF + logistic regression",
        "Word2Vec + feedforward net",
        "Multilingual BERT FP32",
        "Multilingual BERT INT8",
    ]
    for i, (r, label) in enumerate(zip((tf, nn, fp, q), labels)):
        yy = 245 + 46 * i
        if i == 3:
            s.box(46, yy - 29, 868, 43, FAINT)
        row = [
            label,
            f"{r['macro_f1']:.4f}",
            f"{r['accuracy']:.1%}",
            f"{r['size_mb']:.2f}",
            f"{r['mean_ms']:.2f}",
        ]
        for j, val in enumerate(row):
            s.text(xs[j], yy, val, 19, BOLD if i == 3 else SANS, INK, right=j > 0)
        s.rule(46, yy + 15, 868)
    s.text(
        46,
        434,
        "TF-IDF was smallest. Word2Vec was fastest. INT8 had the highest macro-F1 on this test.",
        18,
        BOLD,
    )
    s.text(
        46,
        468,
        "Time includes model preprocessing, CPU inference and the decision. Loading and browser rendering are excluded.",
        15,
        SANS,
        SUB,
    )
    s.end(
        "I gave every model the same five hundred test comments. Accuracy is included for comparison, but macro-F1 is the main quality score because the classes are imbalanced. The fastest model was Word2Vec. INT8 scored highest on this sample. This result could change on a different test set."
    )

    s.heading(
        9,
        "Conclusion",
        "Errors and next steps",
        f"{q['fn']} offensive messages stayed visible",
    )
    s.text(46, 153, "What happened to the 50 offensive comments?", 19, BOLD)
    start, total = 46, 868
    caught = total * q["tp"] / tst["offensive"]
    s.box(start, 174, caught, 16, GREEN)
    s.box(start + caught, 174, total - caught, 16, "#C88469")
    s.text(46, 218, f"{q['tp']} flagged", 18, BOLD, GREEN)
    s.text(914, 218, f"{q['fn']} missed", 18, BOLD, RUST, right=True)
    s.box(46, 246, 868, 119, FAINT)
    s.text(63, 274, "A MISSED TEST COMMENT", 12, BOLD, GREEN)
    s.text(62, 313, "“" + bad["excerpt"] + "”", 28, ITALIC)
    s.text(
        63,
        344,
        f"Dataset label: offensive    /    Model: clean    /    Offensive score: {bad['offensive_score']:.4f}    /    Threshold: {bert_best['threshold']:.3f}",
        16,
        SANS,
        SUB,
    )
    s.text(46, 409, "Cost to players", 17, BOLD)
    s.text(46, 447, "Next test", 17, BOLD)
    s.para(
        225,
        409,
        f"The model also hid {q['fp']} of {tst['clean']} clean comments.",
        682,
        17,
        22,
    )
    s.para(
        225,
        447,
        "Label real gaming chats and test indirect insults with and without the preceding message. The current data comes from Wikipedia discussions.",
        680,
        17,
        22,
    )
    s.end(
        f"This example is from test row {bad['id']}. The model assigned a very low offensive score to an indirectly hostile message. There are only fifty offensive test examples. Before using it in live games, I would collect representative game chat and measure how much conversational context helps."
    )

    s.heading(10, "Conclusion", "Disclosure and references", "AI usage and sources")
    s.box(46, 146, 373, 319, FOREST)
    s.text(67, 183, "AI USAGE", 13, BOLD, MINT)
    s.text(66, 226, "ChatGPT / Codex", 30, SERIF, WHITE)
    s.para(
        68,
        266,
        "I used OpenAI ChatGPT and Codex to help write and debug code, prepare the experiments, and draft the slides and demo narration.",
        325,
        18,
        24,
        color=WHITE,
    )
    s.rule(68, 352, 325, "#52796D")
    s.para(
        68,
        384,
        "The training logs, predictions, file sizes and timings came from running this project.",
        321,
        17,
        24,
        color=LIGHT,
    )
    s.text(457, 168, "Sources and experiment files", 21, BOLD)
    sources = [
        (
            "Jigsaw / Conversation AI",
            "Toxic Comment Classification Challenge",
            "https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge",
        ),
        (
            "Heliosoph",
            "Mirror of the labeled Jigsaw training data",
            "https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments",
        ),
        (
            "Google BERT",
            "Pretrained multilingual cased checkpoint",
            "https://huggingface.co/google-bert/bert-base-multilingual-cased",
        ),
        (
            "PyTorch",
            "Dynamic quantization documentation",
            "https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html",
        ),
        (
            "SentinelAI",
            "Test predictions, run logs and CPU timings",
            repo.rstrip("/") + "/tree/main/runs/final/results",
        ),
    ]
    for i, (name, title, url) in enumerate(sources):
        y = 207 + 56 * i
        s.text(457, y, name, 17, BOLD, GREEN, link=url)
        s.text(457, y + 21, title, 15, SANS, SUB, link=url)
        s.rule(457, y + 32, 456)
    s.end(
        "The linked experiment directory contains the exact test predictions and timing measurements. The dataset, pretrained checkpoint and quantization reference are cited on this slide. My conclusion is specific to this test set and this CPU."
    )
    s.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--powerpoint",
        action="store_true",
        help="Also export a native PowerPoint with presenter notes (requires python-pptx).",
    )
    build(parser.parse_args().powerpoint)
