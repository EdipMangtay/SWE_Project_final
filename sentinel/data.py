import hashlib
from pathlib import Path
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split
from .config import ROOT, SEED, SOURCE, id_hash, save_json
from .text import clean_text

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def prepare(run, mode, raw=None):
    path = Path(raw) if raw else ROOT / "data/raw/train.csv.gz"
    if not path.exists():
        if raw:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(SOURCE, tmp)
        if (
            hashlib.sha256(tmp.read_bytes()).hexdigest()
            != "d27aedebd2e31e3203e667e5d162a17f14ae04a6e55a5894798a7d1902f4bbe5"
        ):
            raise ValueError(
                "Unexpected Jigsaw mirror checksum. Use the official labeled train.csv."
            )
        tmp.replace(path)
    data = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    required = ["id", "comment_text", *LABELS]
    if missing := set(required) - set(data.columns):
        raise ValueError(f"Missing labeled Jigsaw columns: {sorted(missing)}")
    raw_count = len(data)
    valid = data[LABELS].isin([0, 1]).all(axis=1) & data.id.str.strip().ne("")
    data = data.loc[valid].copy()
    data["key"] = data.comment_text.map(clean_text).str.casefold()
    data = data[data.key.ne("")].copy()
    if data.id.duplicated().any():
        raise ValueError("Repeated Jigsaw IDs. Use the original training file.")
    data["label"] = data[LABELS].max(axis=1).astype(int)
    conflicts = data.groupby("key").label.nunique()
    bad = conflicts[conflicts > 1].index
    data = data.loc[~data.key.isin(bad)].drop_duplicates("key").sort_values("id")
    sample_n = 5000 if mode == "final" else 300
    sample, _ = train_test_split(
        data, train_size=sample_n, stratify=data.label, random_state=SEED
    )
    train, rest = train_test_split(
        sample, test_size=0.2, stratify=sample.label, random_state=SEED
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest.label, random_state=SEED
    )
    summary = {
        "mode": mode,
        "source": SOURCE,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw": raw_count,
        "eligible": len(data),
        "eligible_offensive": int(data.label.sum()),
        "conflicting_text_groups": len(bad),
        "sample": len(sample),
        "seed": SEED,
        "splits": {},
    }
    folder = run / "data"
    folder.mkdir(parents=True, exist_ok=True)
    for key, frame in [("train", train), ("validation", val), ("test", test)]:
        frame = frame.sort_values("id")
        frame[["id", "comment_text", "label"]].to_csv(
            folder / f"{key}.csv.gz", index=False
        )
        ids = frame.id.tolist()
        save_json(run / "results" / f"{key}_ids.json", ids)
        positive = int(frame.label.sum())
        summary["splits"][key] = {
            "rows": len(frame),
            "clean": len(frame) - positive,
            "offensive": positive,
            "offensive_pct": 100 * positive / len(frame),
            "mean_chars": float(frame.comment_text.str.len().mean()),
            "id_sha256": id_hash(ids),
        }
    save_json(run / "results/dataset.json", summary)
    return summary


def split(run, name):
    return pd.read_csv(
        run / "data" / f"{name}.csv.gz", dtype={"id": str}, keep_default_na=False
    )
