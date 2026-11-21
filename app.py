from html import escape

import streamlit as st
import torch
from sentinel.config import FINAL, LIMIT, NAMES, ROOT, THREADS
from sentinel.model import load_model
from sentinel.text import lexicon_hit, validate

st.set_page_config(page_title="SentinelAI · Chat moderation", layout="wide")
stylesheet = (ROOT / "sentinel/moderation.css").read_text(encoding="utf-8")
st.html(f"<style>{stylesheet}</style>")
torch.set_num_threads(THREADS)
INFO = {
    "int8": "INT8 linear layers and word embeddings. CPU inference.",
    "bert": "Full-precision multilingual BERT. CPU inference.",
    "tfidf": "Word unigrams and bigrams with logistic regression.",
    "neural": "A feedforward network using mean Word2Vec embeddings.",
}


@st.cache_resource(show_spinner=False)
def cached(kind):
    return load_model(FINAL, kind)


def fetch(kind):
    try:
        return cached(kind), None
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        return (
            None,
            f"{NAMES[kind]} is unavailable. Install its artifacts as described in the README. ({type(exc).__name__})",
        )


def reset_chat():
    st.session_state.history = []
    st.session_state.last_input = None
    st.session_state.comparison = None


if "history" not in st.session_state:
    reset_chat()


def compare_models(text):
    result = []
    for key, name in NAMES.items():
        row = {
            "Model": name,
            "Decision": "Unavailable",
            "Confidence": "—",
            "CPU time": "—",
        }
        adapter, error = fetch(key)
        if not error:
            try:
                prediction = adapter.predict(text)
                row.update(
                    {
                        "Decision": prediction["label"],
                        "Confidence": f"{prediction['confidence']:.1%}",
                        "CPU time": f"{prediction['latency_ms']:.2f} ms",
                    }
                )
            except (ValueError, RuntimeError) as exc:
                st.error(f"{name}: {exc}")
        result.append(row)
    return result


def decision_badge(decision):
    tone = {"CLEAN": "clean", "OFFENSIVE": "flagged"}.get(decision, "quiet")
    return f'<span class="verdict {tone}">{escape(decision)}</span>'


with st.sidebar:
    st.html(
        '<div class="brand"><i class="brand-icon" aria-hidden="true"></i>SentinelAI</div>'
    )
    model_key = st.selectbox("Active model", list(NAMES), format_func=NAMES.get)
    st.html(f'<p class="side-note">{INFO[model_key]}</p>')
    st.divider()
    session_stats = st.container()
    st.button("Clear session", on_click=reset_chat, icon=":material/restart_alt:")

with st.spinner("Opening the selected model…"):
    model, error = fetch(model_key)
status = "Local CPU ready" if model is not None else "Model unavailable"
st.html(f"""
    <header class="masthead">
      <div><h1>Chat moderation</h1><p>Review English game chat with four local classifiers.</p></div>
      <div class="runtime"><i class="{"online" if model is not None else "offline"}"></i>{status}</div>
    </header>""")
if error:
    st.error(error)

with st.container(key="desk"):
    left, right = st.columns([1.05, 1], gap="small")
with left, st.container(key="message_panel"):
    st.html('<div class="panel-title"><h2>Message review</h2></div>')
    feed = st.container()
    message = st.chat_input("Type a message…", max_chars=LIMIT, disabled=model is None)
    if message and model is not None:
        try:
            original = validate(message)
            row = model.predict(original)
            st.session_state.history.append(row)
            st.session_state.last_input = original
            st.session_state.comparison = None
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))

rows = st.session_state.history
with feed:
    if rows:
        with st.container(
            height=337 if len(rows) > 3 else "content", border=False, autoscroll=True
        ):
            for number, row in enumerate(rows, 1):
                tone = "masked" if row["label"] == "OFFENSIVE" else ""
                st.html(f"""
                    <article class="message {tone}" aria-label="Message {number}">
                      <div class="message-top"><span>Message {number:02d}</span>{decision_badge(row["label"])}</div>
                      <div class="message-body">{escape(row["display"])}</div>
                      <div class="message-model">{escape(NAMES[row["model"]])}</div>
                      <div class="message-readings"><span><b>{row["confidence"]:.1%}</b> confidence</span>
                      <span><b>{row["latency_ms"]:.2f} ms</b> CPU time</span></div>
                    </article>""")
    else:
        st.html("""
          <div class="empty-feed"><i class="bubble" aria-hidden="true"></i>
            <strong>Send your first message</strong>
            <p>Its decision, confidence and CPU time will appear here.</p>
          </div>""")

with right, st.container(key="comparison_panel"):
    tag = f"Message {len(rows):02d}" if rows else "Awaiting input"
    st.html(
        f'<div class="panel-title"><h2>Model comparison</h2><span class="tag">{tag}</span></div>'
    )
    if st.button(
        "Compare all four models",
        type="primary",
        icon=":material/compare_arrows:",
        disabled=st.session_state.last_input is None,
        width="stretch",
    ):
        with st.spinner("Comparing models…"):
            st.session_state.comparison = compare_models(st.session_state.last_input)
    result = st.session_state.comparison
    if result is None:
        st.html(
            '<p class="comparison-note">Compare each model’s decision on the latest message.</p>'
        )
    else:
        decisions = [row["Decision"] for row in result]
        flagged_count = decisions.count("OFFENSIVE")
        clean_count = decisions.count("CLEAN")
        counts = [f"{flagged_count} flagged", f"{clean_count} clean"]
        if "Unavailable" in decisions:
            counts.append(f"{decisions.count('Unavailable')} unavailable")
        st.html(f'<p class="comparison-note result-count">{" · ".join(counts)}</p>')
    model_rows = []
    by_model = {row["Model"]: row for row in (result or [])}
    for key, name in NAMES.items():
        if result is None:
            values = f'<span class="pending">{escape(INFO[key])}</span>'
        elif name not in by_model:
            values = decision_badge("Unavailable")
        else:
            entry = by_model[name]
            values = (
                f"{decision_badge(entry['Decision'])}"
                f'<span class="pair">{escape(entry["Confidence"])} · {escape(entry["CPU time"])}</span>'
            )
        model_rows.append(
            f"<li><strong>{escape(name)}</strong><div>{values}</div></li>"
        )
    st.html('<ul class="model-list">' + "".join(model_rows) + "</ul>")
    if result is not None:
        rule_label = (
            "OFFENSIVE" if lexicon_hit(st.session_state.last_input) else "CLEAN"
        )
        st.html(
            f'<div class="rule"><span>Edit-distance baseline</span>{decision_badge(rule_label)}</div>'
        )
        st.caption(
            "Each row shows the decision, selected-class confidence and CPU time."
        )

with session_stats:
    st.html('<h2 class="side-title">This session</h2>')
    flagged = sum(row["label"] == "OFFENSIVE" for row in rows)
    mean = (
        f"{sum(row['latency_ms'] for row in rows) / len(rows):.2f} ms" if rows else "—"
    )
    st.html(f"""
        <div class="session-metrics">
          <div><span>Reviewed</span><strong>{len(rows)}</strong></div>
          <div><span>Flagged</span><strong>{flagged}</strong></div>
        </div><div class="mean"><span>Mean latency</span><b>{mean}</b></div>""")
    if rows:
        with st.expander("Response time log"):
            st.dataframe(
                [
                    {
                        "Message": i,
                        "CPU time (ms)": round(r["latency_ms"], 2),
                        "Model": NAMES[r["model"]],
                    }
                    for i, r in enumerate(rows, 1)
                ],
                hide_index=True,
            )
