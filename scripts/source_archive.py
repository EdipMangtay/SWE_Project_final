"""Create the GitHub-ready source tree without datasets, environments or weights."""

from zipfile import ZIP_DEFLATED, ZipFile
from sentinel.config import ROOT


def source_files():
    for name in [
        "README.md",
        "BASLANGIC.md",
        "Baslat.command",
        "app.py",
        "requirements.txt",
        ".gitignore",
        ".streamlit/config.toml",
    ]:
        yield ROOT / name
    for name in ["sentinel", "scripts", "tests", "docs", "runs/final/results"]:
        for p in sorted((ROOT / name).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store":
                yield p


def main():
    dest = ROOT / "delivery/SentinelAI_GitHub.zip"
    dest.parent.mkdir(exist_ok=True)
    with ZipFile(dest, "w", ZIP_DEFLATED) as out:
        for f in source_files():
            out.write(f, f.relative_to(ROOT))
    print(dest)


if __name__ == "__main__":
    main()
