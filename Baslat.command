#!/bin/zsh
set -e
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  python3.12 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
if [[ ! -s runs/final/artifacts/tfidf/pipeline.joblib || ! -s runs/final/artifacts/neural/vectors.npz ]]; then
  .venv/bin/python -m scripts.train_all --mode final
elif [[ ! -s runs/final/artifacts/bert/model.safetensors || ! -s runs/final/artifacts/int8/int8.pt ]]; then
  .venv/bin/python -m scripts.restore_models
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.showEmailPrompt false
