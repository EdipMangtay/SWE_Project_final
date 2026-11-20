from pathlib import Path
import hashlib
import json
import os
import random

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LIMIT = 4096
TOKENS = 64
THREADS = 4
BUDGET_MS = 50
SOURCE = "https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/resolve/78e67fb081bc05846a90e944d7bb8f2f4d36e438/train.csv.gz"
BASE = "google-bert/bert-base-multilingual-cased"
REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
NAMES = {
    "int8": "Multilingual BERT · INT8",
    "bert": "Multilingual BERT · FP32",
    "tfidf": "TF-IDF + logistic regression",
    "neural": "Word2Vec + neural network",
}
FINAL = ROOT / "runs" / "final"


def seed():
    import torch
    import numpy as np

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def id_hash(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()
