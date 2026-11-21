"""Train on the training split and use validation only for model selection."""

from pathlib import Path
import tempfile

from collections import Counter
import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification, BertTokenizer
from .config import BASE, REVISION, SEED, TOKENS, save_json, seed
from .data import split
from .model import (
    FeedForward,
    sentence_vectors,
    encoding,
    load_model,
    quantize,
    quantized_layers,
)
from .text import tokens


def threshold_for(y, scores):
    candidates = np.arange(0.10, 0.901, 0.025)
    results = [
        (f1_score(y, scores >= t, average="macro", zero_division=0), -abs(t - 0.5), t)
        for t in candidates
    ]
    f1, _, threshold = max(results)
    return float(threshold), float(f1)


def directory(run, kind):
    dest = run / "artifacts" / kind
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def classical(run):
    train, val = split(run, "train"), split(run, "validation")
    folder = directory(run, "tfidf")
    best = -1.0
    history = []
    for c in (1.0, 4.0):
        model = Pipeline(
            [
                (
                    "words",
                    TfidfVectorizer(
                        tokenizer=tokens,
                        token_pattern=None,
                        lowercase=False,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=35000,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c, class_weight="balanced", max_iter=400, random_state=SEED
                    ),
                ),
            ]
        )
        model.fit(train.comment_text, train.label)
        threshold, f1 = threshold_for(
            val.label, model.predict_proba(val.comment_text)[:, 1]
        )
        history.append({"C": c, "validation_macro_f1": f1, "threshold": threshold})
        if f1 > best:
            best = f1
            joblib.dump(model, folder / "pipeline.joblib", compress=3)
            save_json(folder / "metadata.json", {"threshold": threshold, "C": c})
    save_json(run / "results/tfidf_training.json", history)


def train_embeddings(texts, epochs):
    # Skip-gram with negative sampling. Source and context never cross a message boundary.
    corpus = [tokens(s)[:128] for s in texts]
    frequencies = Counter(w for sentence in corpus for w in sentence)
    words = sorted(w for w, count in frequencies.items() if count >= 2)
    vocab = {w: i for i, w in enumerate(words)}
    pairs = []
    rng = np.random.default_rng(SEED)
    total = sum(frequencies.values())
    for sentence in corpus:
        ids = [
            vocab[w]
            for w in sentence
            if w in vocab
            and rng.random()
            <= min(
                1.0,
                (1.0e-3 * total / frequencies[w]) ** 0.5
                + 1.0e-3 * total / frequencies[w],
            )
        ]
        for i, word in enumerate(ids):
            pairs.extend(
                (word, ids[j])
                for j in range(max(0, i - 2), min(len(ids), i + 3))
                if j != i
            )
    pairs = torch.tensor(pairs)
    distribution = torch.tensor(
        [frequencies[w] ** 0.75 for w in words], dtype=torch.float32
    )
    center = nn.Embedding(len(words), 64, sparse=True)
    context = nn.Embedding(len(words), 64, sparse=True)
    nn.init.uniform_(center.weight, -0.5 / 64, 0.5 / 64)
    nn.init.zeros_(context.weight)
    optimizer = torch.optim.SparseAdam([center.weight, context.weight], lr=0.012)
    losses = []
    for _ in range(epochs):
        total_loss = 0.0
        for index in torch.randperm(len(pairs)).split(1024):
            source, target = pairs[index].T
            # Draw five negatives per positive pair from unigram^0.75.
            neg = torch.multinomial(
                distribution, len(index) * 5, replacement=True
            ).view(-1, 5)
            v = center(source)
            loss = -(
                F.logsigmoid((v * context(target)).sum(1))
                + F.logsigmoid(-(v[:, None, :] * context(neg)).sum(-1)).sum(1)
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)
        losses.append(total_loss / len(pairs))
    return (
        words,
        center.weight.detach().numpy(),
        {"pairs": len(pairs), "word2vec_loss": losses},
    )


def neural(run, mode):
    seed()
    tr, va = split(run, "train"), split(run, "validation")
    folder = directory(run, "neural")
    words, vectors, log = train_embeddings(tr.comment_text, 4 if mode == "final" else 1)
    np.savez_compressed(folder / "vectors.npz", words=np.array(words), vectors=vectors)
    vocab = {w: i for i, w in enumerate(words)}
    x, z = (sentence_vectors(df.comment_text, vocab, vectors) for df in (tr, va))
    y, v = (
        torch.tensor(df.label.values.copy(), dtype=torch.float32) for df in (tr, va)
    )
    model = FeedForward(vectors.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    criterion = nn.BCEWithLogitsLoss(pos_weight=(1 - y).sum() / y.sum())
    best, wait = -1.0, 0
    log["epochs"] = []
    for epoch in range(24 if mode == "final" else 2):
        model.train()
        total = 0.0
        for ix in torch.randperm(len(y)).split(64):
            loss = criterion(model(x[ix]).flatten(), y[ix])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        model.eval()
        with torch.no_grad():
            logits = model(z).flatten()
            threshold, f1 = threshold_for(v.numpy(), logits.sigmoid().numpy())
            val_loss = float(criterion(logits, v))
        log["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(y),
                "val_loss": val_loss,
                "val_macro_f1": f1,
            }
        )
        if f1 > best:
            best, wait = f1, 0
            torch.save(model.state_dict(), folder / "network.pt")
            save_json(
                folder / "metadata.json",
                {
                    "threshold": threshold,
                    "best_epoch": epoch + 1,
                    "dim": 64,
                    "hidden": 64,
                    "dropout": 0.2,
                    "vocab_size": len(words),
                },
            )
        else:
            wait += 1
            if wait == 5:
                break
    save_json(run / "results/neural_training.json", log)


def transformer(run, mode, checkpoint=None):
    seed()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    tr, va = split(run, "train"), split(run, "validation")
    local = checkpoint is not None
    source = str(checkpoint) if local else BASE
    kwargs = {"local_files_only": True} if local else {"revision": REVISION}
    tok = BertTokenizer.from_pretrained(source, **kwargs)
    model = BertForSequenceClassification.from_pretrained(
        source, num_labels=2, attn_implementation="eager", **kwargs
    )
    if model.config.vocab_size != 119547 or model.config.num_hidden_layers != 12:
        raise ValueError("Expected the pinned 12-layer multilingual cased BERT.")
    model.config.id2label = {0: "CLEAN", 1: "OFFENSIVE"}
    model.config.label2id = {"CLEAN": 0, "OFFENSIVE": 1}
    for module in [model.bert.embeddings, *model.bert.encoder.layer[:-2]]:
        module.requires_grad_(False)
    model.to(device)
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=4e-5
    )
    y = torch.tensor(tr.label.to_numpy().copy(), device=device)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, float(((1 - y).sum() / y.sum()).sqrt())], device=device
        )
    )
    texts = tr.comment_text.tolist()
    folder = directory(run, "bert")
    log = {
        "base": BASE,
        "revision": REVISION,
        "device": device,
        "batch_size": 16,
        "tokens": TOKENS,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "frozen_lower_blocks": 10,
        "optimizer": "AdamW",
        "learning_rate": 4e-5,
        "epochs": [],
    }
    history = run / "results/bert_training.json"
    best = -1.0
    for epoch in range(3 if mode == "final" else 1):
        model.train()
        # Feature extraction from frozen blocks remains deterministic.
        model.bert.embeddings.eval()
        for layer in model.bert.encoder.layer[:-2]:
            layer.eval()
        total = 0.0
        for step, ix in enumerate(torch.randperm(len(y)).split(16)):
            batch = encoding(tok, [texts[i] for i in ix.tolist()], device)
            loss = loss_fn(model(**batch).logits, y[ix.to(device)])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += float(loss.detach()) * len(ix)
            if step % 50 == 0:
                print(
                    f"BERT epoch {epoch + 1} | batch {step + 1} | loss {float(loss.detach()):.4f}",
                    flush=True,
                )
        model.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(va), 16):
                batch = encoding(tok, va.comment_text.iloc[start : start + 16], device)
                scores.extend(model(**batch).logits.softmax(-1)[:, 1].cpu().tolist())
        threshold, f1 = threshold_for(va.label, np.asarray(scores))
        log["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(y),
                "val_macro_f1": f1,
                "threshold": threshold,
            }
        )
        if f1 > best:
            best = f1
            tok.save_pretrained(folder)
            model.save_pretrained(folder)
            save_json(
                folder / "metadata.json",
                {
                    "threshold": threshold,
                    "best_epoch": epoch + 1,
                    "source": BASE,
                    "revision": REVISION,
                },
            )
        save_json(history, log)
        print(log["epochs"][-1], flush=True)


def export_int8(run):
    seed()
    fp = load_model(run, "bert")
    folder = directory(run, "int8")
    fp.model.config.save_pretrained(folder)
    fp.tokenizer.save_pretrained(folder)
    model = quantize(fp.model)
    count = quantized_layers(model)
    if count != 74:
        raise ValueError(
            f"Expected 74 dynamically quantized BERT linear layers, found {count}."
        )
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(model.state_dict(), temp)
        temp.replace(folder / "int8.pt")
    finally:
        temp.unlink(missing_ok=True)
    save_json(
        folder / "metadata.json",
        {
            "threshold": fp.threshold,
            "linear_modules": count,
            "word_embedding": "symmetric per-row INT8",
            "engine_at_export": torch.backends.quantized.engine,
        },
    )
