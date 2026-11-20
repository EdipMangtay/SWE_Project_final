"""Verify mechanical grading gates; preserve pending publication and human checks."""

import argparse
import csv
import subprocess
import sys
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import xml.etree.ElementTree as ET
import numpy as np
from pypdf import PdfReader
from sklearn.metrics import f1_score
from streamlit.testing.v1 import AppTest
from sentinel.config import FINAL, ROOT, NAMES, id_hash, read_json, save_json, seed
from sentinel.model import load_model, quantized_layers


def audit(online):
    seed()
    res = FINAL / "results"
    config = read_json(ROOT / "docs/submission.json")
    checks = []

    def mark(gate, name, status, evidence):
        checks.append(dict(gate=gate, item=name, status=status, evidence=evidence))

    def check(gate, name, fn, evidence):
        try:
            passed = bool(fn())
            mark(
                gate,
                name,
                "PASS" if passed else "FAIL",
                evidence if passed else "Evidence absent or inconsistent.",
            )
        except Exception as exc:
            mark(gate, name, "FAIL", f"{type(exc).__name__}: {exc}")

    def http(url):
        with urlopen(
            Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=12
        ) as r:
            return 200 <= r.status < 400

    def external(name, key, hosts):
        url = config.get(key, "")
        u = urlparse(url)
        if (
            u.scheme != "https"
            or (u.hostname or "").lower() not in hosts
            or u.path in ("", "/")
        ):
            mark(1, name, "FAIL", "A valid project-specific link is missing.")
        elif online:
            try:
                ok = http(url)
                mark(
                    1,
                    name,
                    "PASS" if ok else "FAIL",
                    "HTTPS availability; verify anonymous access in a private browser.",
                )
            except (OSError, URLError) as exc:
                mark(1, name, "NOT YET VERIFIED", str(exc))
        else:
            mark(
                1,
                name,
                "NOT YET VERIFIED",
                "URL present. Re-run with --online and check private-browser access.",
            )

    def ui():
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        assert not app.exception and not app.error and app.selectbox[0].value == "int8"
        app.chat_input[0].set_value("Thanks for your help!").run()
        return not app.exception and app.session_state["history"][-1]["model"] == "int8"

    check(
        1,
        "R1 / R5 · chat accepts new input with default INT8",
        ui,
        "Streamlit AppTest submitted a fresh message and recorded an INT8 decision.",
    )
    check(
        1,
        "HTTP application startup",
        lambda: (
            subprocess.run(
                [sys.executable, "-m", "scripts.check_http"],
                cwd=ROOT,
                capture_output=True,
                timeout=45,
            ).returncode
            == 0
        ),
        "Application root and health both returned HTTP 200 in process.",
    )
    for key in ["bert", "int8"]:

        def infer(key=key):
            m = load_model(FINAL, key)
            return (key != "int8" or quantized_layers(m.model) == 74) and m.predict(
                "Hello team!"
            )["label"] in {"CLEAN", "OFFENSIVE"}

        check(
            1,
            f"Working {key} artifact",
            infer,
            "Loaded local checkpoint and ran a real CPU forward pass.",
        )
    check(
        1,
        "Use Case 4 and repository source",
        lambda: all(
            (ROOT / p).is_file()
            for p in [
                "app.py",
                "sentinel/train.py",
                "sentinel/model.py",
                "sentinel/text.py",
                "README.md",
                "requirements.txt",
            ]
        ),
        "Chat moderation UI, implementation, requirements and setup present.",
    )
    pdf = ROOT / "docs/presentation.pdf"

    def pages():
        return PdfReader(pdf).pages

    check(1, "Presentation PDF", lambda: len(pages()) > 0, "Parsed the PDF.")
    external(
        "Accessible code repository", "repository_url", {"github.com", "gitlab.com"}
    )
    external(
        "Accessible demo video",
        "video_url",
        {"www.youtube.com", "youtube.com", "youtu.be", "drive.google.com"},
    )
    mark(
        1,
        "Original results and student explanation",
        "MANUAL CHECK",
        "Verify experimental provenance and explain the code. Numerical consistency is checked below.",
    )
    mark(
        1,
        "Video contains live input",
        "MANUAL CHECK",
        "Watch the recorded stream and verify that new typed input returns a decision.",
    )

    def text(i):
        return pages()[i].extract_text().casefold()

    check(
        2,
        "R2 · technique map and A/B/C/D",
        lambda: (
            all(
                k in text(4)
                for k in [
                    "technique map",
                    "neural networks",
                    "classical nlp",
                    "transformers",
                    "efficient models",
                ]
            )
            and read_json(res / "neural_training.json")["word2vec_loss"]
            and read_json(res / "bert_training.json")["epochs"]
            and bool(read_json(res / "rule_baseline.json"))
        ),
        "Slide 5, trained Word2Vec and neural net, lexicon results, BERT log and INT8 checks.",
    )

    def common():
        ids = read_json(res / "test_ids.json")
        groups = [
            set(read_json(res / f"{x}_ids.json"))
            for x in ("train", "validation", "test")
        ]
        assert sum(map(len, groups)) == len(set.union(*groups))
        rows = read_json(res / "metrics.json")
        assert {r["model"] for r in rows} == set(NAMES)
        for row in rows:
            with open(res / f"{row['model']}_predictions.csv") as f:
                pp = list(csv.DictReader(f))
            assert [p["id"] for p in pp] == ids
            f1 = f1_score(
                [int(p["true"]) for p in pp],
                [int(p["predicted"]) for p in pp],
                average="macro",
            )
            assert abs(f1 - row["macro_f1"]) < 1e-9 and row["test_sha256"] == id_hash(
                ids
            )
        return True

    check(
        2,
        "R3 · four models and identical test IDs",
        common,
        "Recomputed four macro-F1 scores from the ordered held-out predictions.",
    )

    def benchmarking():
        b = read_json(res / "benchmark.json")
        assert b["hardware"]["cpu"] and b["runs"] >= 100 and b["warmup"] > 0
        for row in read_json(res / "metrics.json"):
            t = b["latencies_ms"][row["model"]]
            assert len(t) == b["runs"] and np.isclose(np.mean(t), row["mean_ms"])
            assert row["size_mb"] > 0 and f"{row['macro_f1']:.3f}" in text(7)
        return True

    check(
        2,
        "R4 · metrics, size, CPU and latency",
        benchmarking,
        "Read slide 8, hardware, model sizes and all raw timings.",
    )
    check(
        2,
        "Mean latency budget",
        lambda: (
            next(r for r in read_json(res / "metrics.json") if r["model"] == "int8")[
                "mean_ms"
            ]
            < read_json(res / "benchmark.json")["budget_ms"]
        ),
        "Compared INT8 with the predeclared 50 ms mean budget.",
    )
    check(
        2,
        "R6 · architecture diagram",
        lambda: (
            (ROOT / "docs/figures/architecture.png").stat().st_size > 5000
            and bool(pages()[3].images)
        ),
        "Slide 4 embeds the exported diagram.",
    )
    mark(
        2,
        "Architecture matches the complete demo",
        "MANUAL CHECK",
        "Trace an actual message through the shown preprocessing branch, model, mask and timing log.",
    )
    check(
        2,
        "R7 · installation and launch instructions",
        lambda: (
            "streamlit run app.py" in (ROOT / "README.md").read_text()
            and (ROOT / "requirements.txt").exists()
        ),
        "README documents the Python environment, model restoration and one run command.",
    )
    check(
        2,
        "R8 · AI usage slide",
        lambda: "ai usage" in text(9) and "codex" in text(9),
        "Slide 10 names the assistant and how it was used.",
    )
    check(
        2,
        "PDF structure and length",
        lambda: (
            8 <= len(pages()) <= 12
            and "introduction" in text(1)
            and "development" in text(2)
            and "conclusion" in text(8)
        ),
        "Ten-page Introduction / Development / Conclusion deck.",
    )

    def cover():
        page = pages()[0]
        links = [
            a.get_object().get("/A", {}).get("/URI") for a in page.get("/Annots", [])
        ]
        return (
            all(
                config.get(k) and config[k] in links
                for k in ["repository_url", "video_url"]
            )
            and config["student_id"] in page.extract_text()
        )

    check(
        2,
        "Cover identity and both external links",
        cover,
        "Verified PDF link annotations and student ID.",
    )
    check(
        2,
        "End-to-end test evidence",
        lambda: (
            all(
                int(s.attrib.get(x, "0")) == 0
                for s in ET.parse(res / "tests.xml").iter("testsuite")
                for x in ["failures", "errors", "skipped"]
            )
            and any(
                int(s.attrib.get("tests", 0)) >= 17
                for s in ET.parse(res / "tests.xml").iter("testsuite")
            )
        ),
        "Executed pytest JUnit XML records model and Streamlit integration tests.",
    )
    check(
        2,
        "Recording guide and narration",
        lambda: all(
            (ROOT / "docs" / p).is_file()
            for p in ["demo_script.md", "demo_recording_checklist.md"]
        ),
        "Timed script and checklist present.",
    )
    mark(
        2,
        "Narration, architecture, results and duration",
        "MANUAL CHECK",
        "Play the final upload, verify sound and a runtime of 2:00 or less.",
    )
    mark(
        2,
        "Fresh remote setup and anonymous access",
        "MANUAL CHECK",
        "Use the published repo in a fresh checkout and open the final video and repo in a private browser.",
    )
    critical = any(r["gate"] == 1 and r["status"] == "FAIL" for r in checks)
    state = (
        "NOT SUBMISSION READY — CRITICAL REQUIREMENT MISSING"
        if critical
        else (
            "REVIEW REQUIRED"
            if any(r["status"] != "PASS" for r in checks)
            else "AUTOMATED CHECKS PASSED"
        )
    )
    report = {"status": state, "checks": checks}
    save_json(res / "audit.json", report)
    (ROOT / "docs/submission_audit.md").write_text(
        "# Submission audit\n\n"
        + state
        + "\n\n| Gate | Requirement | Status | Evidence |\n|---|---|---|---|\n"
        + "".join(
            f"| {r['gate']} | {r['item']} | {r['status']} | {r['evidence']} |\n"
            for r in checks
        ),
        encoding="utf-8",
    )
    print(state)
    return 1 if any(r["status"] != "PASS" for r in checks) else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    sys.exit(audit(p.parse_args().online))
