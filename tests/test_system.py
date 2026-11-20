import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from sentinel.config import FINAL, NAMES, ROOT, id_hash, read_json, seed
from sentinel.model import load_model, quantized_layers
from sentinel.text import displayed, edit_distance, lexicon_hit, tokens, validate


@pytest.fixture(scope="module", autouse=True)
def cpu():
    seed()


@pytest.mark.parametrize("text", ["", "   ", None, "x" * 4097])
def test_rejected_input(text):
    with pytest.raises(ValueError):
        validate(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("h3llo    team", ["hello", "team"]),
        ("soooo cool 2026", ["soo", "cool", "2026"]),
        ("İYİ ıŞIK", ["iyi", "işik"]),
    ],
)
def test_tokens(text, expected):
    assert tokens(text) == expected


def test_edit_distance_and_safe_display():
    assert edit_distance("idiot", "idiotx") == 1 and lexicon_hit("id1otx")
    assert not lexicon_hit("Good game, friend")
    assert displayed("<tag>", False) == "<tag>"
    assert displayed("secret text", True) == "••••  Message hidden"


def test_split_ids_and_no_overlap():
    res = FINAL / "results"
    groups = []
    for name in ["train", "validation", "test"]:
        ids = read_json(res / f"{name}_ids.json")
        assert len(set(ids)) == len(ids)
        groups.append(set(ids))
        assert read_json(res / "dataset.json")["splits"][name]["id_sha256"] == id_hash(
            ids
        )
    assert all(groups[a].isdisjoint(groups[b]) for a, b in [(0, 1), (1, 2), (0, 2)])


@pytest.mark.parametrize("name", NAMES)
def test_real_adapter(name):
    model = load_model(FINAL, name)
    if name == "int8":
        assert quantized_layers(model.model) == 74
        assert (
            str(model.model.bert.embeddings.word_embeddings.codes.dtype) == "torch.int8"
        )
    for msg in [
        "Good game, thanks for playing.",
        "You are a stupid idiot.",
        "🎮 ✨",
        "ok " * 1300,
    ]:
        pred = model.predict(msg)
        assert pred["label"] in ("CLEAN", "OFFENSIVE") and pred["latency_ms"] > 0
        assert 0 <= pred["confidence"] <= 1 and 0 <= pred["offensive_score"] <= 1
        assert pred["display"] == displayed(msg.strip(), pred["label"] == "OFFENSIVE")


def test_fresh_process_int8():
    p = subprocess.run(
        [sys.executable, "-m", "scripts.check_int8"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_exports_match_predictions():
    from sklearn.metrics import f1_score

    res = FINAL / "results"
    bench = read_json(res / "benchmark.json")
    rows = read_json(res / "metrics.json")
    assert {r["model"] for r in rows} == set(NAMES)
    assert bench["batch_size"] == 1 and len(bench["timed_ids"]) >= 100
    for r in rows:
        preds = pd.read_csv(res / f"{r['model']}_predictions.csv", dtype={"id": str})
        assert preds.id.tolist() == read_json(res / "test_ids.json")
        assert r["macro_f1"] == pytest.approx(
            f1_score(preds.true, preds.predicted, average="macro")
        )
        assert r["mean_ms"] == pytest.approx(np.mean(bench["latencies_ms"][r["model"]]))
        assert r["size_mb"] > 0


def test_app_default_chat_compare_switch_clear():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception and not app.error
    assert app.selectbox[0].value == "int8"
    for msg in ["Thanks for your help!", "You are a stupid idiot."]:
        app.chat_input[0].set_value(msg).run()
    assert not app.exception
    history = app.session_state["history"]
    assert [r["model"] for r in history] == ["int8", "int8"]
    assert history[0]["display"] == "Thanks for your help!"
    assert (
        history[1]["label"] == "OFFENSIVE"
        and history[1]["display"] == "••••  Message hidden"
    )
    next(b for b in app.button if b.label == "Compare all four models").click().run()
    assert len(app.session_state["comparison"]) == 4
    assert all(
        r["Decision"] in ("CLEAN", "OFFENSIVE") for r in app.session_state["comparison"]
    )
    app.selectbox[0].select("tfidf").run()
    app.chat_input[0].set_value("Good morning").run()
    assert app.session_state["history"][-1]["model"] == "tfidf"
    next(b for b in app.button if b.label == "Clear session").click().run()
    assert (
        not app.exception
        and app.session_state["history"] == []
        and app.session_state["last_input"] is None
    )


def test_streamlit_http_startup():
    p = subprocess.run(
        [sys.executable, "-m", "scripts.check_http"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert p.returncode == 0, p.stderr
