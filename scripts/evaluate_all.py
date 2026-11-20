"""Final test and single-message CPU benchmark. Run after all training stops."""

import argparse
import gc
import importlib.metadata as md
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sentinel.config import (
    BUDGET_MS,
    LIMIT,
    NAMES,
    ROOT,
    SEED,
    THREADS,
    TOKENS,
    id_hash,
    read_json,
    save_json,
    seed,
)
from sentinel.data import split
from sentinel.model import load_model
from sentinel.text import lexicon_hit


def hardware():
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        info = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True
        ).stdout
        for line in info.splitlines():
            if line.strip().startswith("Chip:"):
                cpu = line.split(":", 1)[1].strip()
    return {
        "cpu": cpu,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch_threads": THREADS,
        "versions": {
            p: md.version(p) for p in ["torch", "transformers", "numpy", "scikit-learn"]
        },
    }


def classify(y, preds):
    matrix = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    return {
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, preds)),
        "offensive_precision": float(precision_score(y, preds, zero_division=0)),
        "offensive_recall": float(recall_score(y, preds, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate(mode):
    seed()
    run = ROOT / "runs" / mode
    res = run / "results"
    df = split(run, "test")
    ids = df.id.tolist()
    if ids != read_json(res / "test_ids.json"):
        raise ValueError("Test IDs differ from the frozen split.")
    strings = [t[:LIMIT] for t in df.comment_text]
    labels = df.label.to_numpy()
    rng = np.random.default_rng(SEED)
    timed = rng.choice(len(df), 150, replace=len(df) < 150).tolist()
    warm = rng.choice(len(df), 15, replace=False).tolist()
    scheme = {
        "hardware": hardware(),
        "mode": mode,
        "seed": SEED,
        "runs": len(timed),
        "warmup": len(warm),
        "budget_ms": BUDGET_MS,
        "batch_size": 1,
        "max_wordpieces": TOKENS,
        "timed_ids": [ids[i] for i in timed],
        "warmup_ids": [ids[i] for i in warm],
        "timing": "Validation-excluded model preprocessing, CPU forward and score decision; UI and loading excluded.",
    }
    rows, timings, failures = [], {}, {}
    for kind in ("tfidf", "neural", "bert", "int8"):
        model = load_model(run, kind)
        predictions = []
        for text in strings:
            predictions.append(model.predict(text))
        pd.DataFrame(
            {
                "id": ids,
                "true": labels,
                "predicted": [int(p["label"] == "OFFENSIVE") for p in predictions],
                "offensive_score": [p["offensive_score"] for p in predictions],
            }
        ).to_csv(res / f"{kind}_predictions.csv", index=False)
        predicted = np.array([int(p["label"] == "OFFENSIVE") for p in predictions])
        record = classify(labels, predicted)
        for index in warm:
            model.predict(strings[index])
        ms = [model.predict(strings[i])["latency_ms"] for i in timed]
        timings[kind] = ms
        folder = run / "artifacts" / kind
        record.update(
            model=kind,
            name=NAMES[kind],
            size_mb=sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
            / 1e6,
            mean_ms=float(np.mean(ms)),
            median_ms=float(np.median(ms)),
            p95_ms=float(np.percentile(ms, 95)),
            test_n=len(df),
            test_sha256=id_hash(ids),
            runs=len(ms),
        )
        failures[kind] = [
            {
                "id": ids[i],
                "true": int(labels[i]),
                "predicted": int(predicted[i]),
                "offensive_score": predictions[i]["offensive_score"],
            }
            for i in np.flatnonzero(labels != predicted)
        ]
        rows.append(record)
        print(kind, record, flush=True)
        del model
        gc.collect()
    misses = [e for e in failures["int8"] if e["true"] == 1]
    if misses:
        lookup = dict(zip(ids, strings))
        preferred = next((e for e in misses if e["id"] == "779c81facf11970b"), None)
        case = preferred or min(misses, key=lambda e: len(lookup[e["id"]]))
        save_json(
            res / "failure_example.json",
            {
                "id": case["id"],
                "excerpt": " ".join(lookup[case["id"]].split())[:140],
                "true": "OFFENSIVE",
                "predicted": "CLEAN",
                "offensive_score": case["offensive_score"],
            },
        )
    scheme["latencies_ms"] = timings
    save_json(res / "benchmark.json", scheme)
    save_json(res / "metrics.json", rows)
    save_json(res / "errors.json", failures)
    save_json(
        res / "rule_baseline.json",
        classify(labels, np.asarray([lexicon_hit(t) for t in strings], dtype=int)),
    )
    pd.DataFrame(rows).drop(columns=["confusion_matrix"]).to_csv(
        res / "comparison.csv", index=False
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dev", "final"], default="final")
    evaluate(p.parse_args().mode)
