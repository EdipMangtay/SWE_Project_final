#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  python3.12 -m venv .venv
fi
if ! .venv/bin/python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
try:
    for line in Path("requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, expected = line.split("==")
            if version(name) != expected:
                raise SystemExit(1)
except PackageNotFoundError:
    raise SystemExit(1)
PY
then
  .venv/bin/python -m ensurepip --upgrade
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -m scripts.restore_models --if-needed
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.showEmailPrompt false
