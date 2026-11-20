import argparse
import subprocess
import sys
from pathlib import Path
from sentinel.config import ROOT, seed
from sentinel.data import prepare
from sentinel.train import classical, neural, transformer, export_int8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "final"], default="final")
    parser.add_argument(
        "--stage",
        choices=["data", "tfidf", "neural", "bert", "int8", "all"],
        default="all",
    )
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    run = ROOT / "runs" / args.mode
    seed()
    stages = {
        "data": lambda: prepare(run, args.mode, args.raw),
        "tfidf": lambda: classical(run),
        "neural": lambda: neural(run, args.mode),
        "bert": lambda: transformer(run, args.mode, args.checkpoint),
        "int8": lambda: export_int8(run),
    }
    for name, step in stages.items():
        if args.stage in (name, "all"):
            print(f"Starting {name} ({args.mode})", flush=True)
            step()
            if name == "int8":
                subprocess.run(
                    [sys.executable, "-m", "scripts.check_int8", "--mode", args.mode],
                    cwd=ROOT,
                    check=True,
                )


if __name__ == "__main__":
    main()
