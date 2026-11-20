"""Run the Streamlit HTTP stack in-process, including startup and shutdown."""

from starlette.testclient import TestClient
from streamlit.web.server.starlette.starlette_app import App
from sentinel.config import ROOT

if __name__ == "__main__":
    with TestClient(App(str(ROOT / "app.py"))) as client:
        home = client.get("/")
        health = client.get("/_stcore/health")
        assert home.status_code == 200 and "streamlit" in home.text.lower()
        assert health.status_code == 200 and health.text.strip() == "ok"
        print("GET / 200; GET /_stcore/health 200 ok")
