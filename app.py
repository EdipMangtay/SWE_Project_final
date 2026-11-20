import streamlit as st
import torch
from sentinel.config import FINAL, LIMIT, NAMES, THREADS
from sentinel.model import load_model
from sentinel.text import lexicon_hit, validate

st.set_page_config(page_title="SentinelAI · Chat moderation", layout="wide")
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

with st.sidebar:
    st.header("Moderation settings")
    model_key = st.selectbox("Active model", list(NAMES), format_func=NAMES.get)
    st.caption(INFO[model_key])
    st.button("Clear session", on_click=reset_chat)
    session_stats = st.container()

st.title("SentinelAI")
st.caption("Real-Time Intelligent Chat Moderation")
st.write("Send an English message to see how the active model moderates it.")
with st.spinner("Opening the selected model…"):
    model, error = fetch(model_key)
if error:
    st.error(error)
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

for row in st.session_state.history:
    with st.chat_message("user"):
        st.text(row["display"])
        st.caption(
            f"{row['label']}  ·  {row['confidence']:.1%} confidence  ·  {row['latency_ms']:.2f} ms  ·  {NAMES[row['model']]}"
        )

with session_stats:
    rows = st.session_state.history
    st.divider()
    st.subheader("This session")
    st.write(
        f"{len(rows)} messages · {sum(r['label'] == 'OFFENSIVE' for r in rows)} flagged"
    )
    if rows:
        st.caption(
            f"Mean latency  {sum(r['latency_ms'] for r in rows) / len(rows):.2f} ms"
        )
        st.dataframe(
            [
                {
                    "Message": i + 1,
                    "Model": NAMES[r["model"]],
                    "ms": round(r["latency_ms"], 2),
                }
                for i, r in enumerate(rows)
            ],
            hide_index=True,
        )

if st.session_state.last_input:
    st.divider()
    if st.button("Compare all four models"):
        result = []
        with st.spinner("Scoring the last message with each model…"):
            for key, name in NAMES.items():
                adapter, error = fetch(key)
                if error:
                    result.append(
                        {
                            "Model": name,
                            "Decision": "Unavailable",
                            "Confidence": "—",
                            "CPU time": "—",
                        }
                    )
                    continue
                try:
                    pred = adapter.predict(st.session_state.last_input)
                    result.append(
                        {
                            "Model": name,
                            "Decision": pred["label"],
                            "Confidence": f"{pred['confidence']:.1%}",
                            "CPU time": f"{pred['latency_ms']:.2f} ms",
                        }
                    )
                except (RuntimeError, ValueError) as exc:
                    st.error(f"{name}: {exc}")
        st.session_state.comparison = result
    if st.session_state.comparison is not None:
        st.dataframe(st.session_state.comparison, hide_index=True)
        hit = lexicon_hit(st.session_state.last_input)
        st.caption(
            "Independent edit-distance rule baseline  ·  "
            + ("OFFENSIVE" if hit else "CLEAN")
        )
    st.caption(
        "Confidence is the selected class score. Validation thresholds can select a class below 50%. Session text stays in memory."
    )
