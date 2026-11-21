"""Losslessly restore final FP32 weights from the pinned public base and measured training changes."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from urllib.request import urlopen
import tempfile
import zipfile
from pathlib import Path
from filelock import FileLock
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from sentinel.config import BASE, FINAL, REVISION, ROOT, seed
from sentinel.train import export_int8

BASE_SHA256 = "876f584f15ebf14887dec17539c114bb99a032e96b9e72507a51c41e205337fc"


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tensor_digest(x):
    return hashlib.sha256(x.view(np.uint8)).hexdigest()


def byte_planes(x):
    # Group bytes of equal significance to compress small FP32 updates losslessly.
    return x.reshape(-1).view(np.uint8).reshape(-1, 4).T.copy().tobytes()


def pack(base, target, patch):
    if digest(base) != BASE_SHA256:
        raise ValueError("Pretrained checksum mismatch.")
    manifest = {}
    with (
        safe_open(base, framework="numpy") as ref,
        safe_open(target, framework="numpy") as fine,
        zipfile.ZipFile(
            patch, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as out,
    ):
        keys = set(ref.keys())
        for name in fine.keys():
            arr = fine.get_tensor(name)
            if arr.dtype != np.float32:
                raise ValueError(f"Unexpected dtype for {name}")
            original = ref.get_tensor(name) if name in keys else None
            mode = "base"
            if original is None or not np.array_equal(
                arr.view(np.uint32), original.view(np.uint32)
            ):
                mode = "raw" if original is None else "xor"
                bits = arr.view(np.uint32)
                if original is not None:
                    bits = bits ^ original.view(np.uint32)
                out.writestr(name, byte_planes(bits))
            manifest[name] = {
                "mode": mode,
                "shape": arr.shape,
                "sha256": tensor_digest(arr),
            }
        out.writestr("manifest.json", json.dumps(manifest, sort_keys=True))


def restore(base, patch, dest):
    if digest(base) != BASE_SHA256:
        raise ValueError(
            "Expected the pinned multilingual BERT base. Check the file SHA-256."
        )
    tensors = {}
    with safe_open(base, framework="numpy") as ref, zipfile.ZipFile(patch) as archive:
        for name, entry in json.loads(archive.read("manifest.json")).items():
            if entry["mode"] == "base":
                arr = ref.get_tensor(name)
            else:
                bits = (
                    np.frombuffer(archive.read(name), dtype=np.uint8)
                    .reshape(4, -1)
                    .T.copy()
                )
                arr = bits.view(np.float32).reshape(entry["shape"])
                if entry["mode"] == "xor":
                    arr = (
                        arr.view(np.uint32) ^ ref.get_tensor(name).view(np.uint32)
                    ).view(np.float32)
            if tensor_digest(arr) != entry["sha256"]:
                raise ValueError(f"Restored tensor did not match the final run: {name}")
            tensors[name] = arr
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=dest.parent, suffix=".safetensors", delete=False
        ) as f:
            temp = Path(f.name)
        try:
            save_file(tensors, temp, metadata={"format": "pt"})
            temp.replace(dest)
        finally:
            temp.unlink(missing_ok=True)


# The companion is data for this exact trained run. Large base weights come from Hugging Face.
COMPANION_URL = "https://github.com/EdipMangtay/SWE_Project_final/releases/download/v1.1.0/SentinelAI_Model_Files.zip"
COMPANION_SHA256 = "60d589aa2263324bf49555020850d7e1882b4c7f35d258c0b463ad010ed75959"
ARTIFACT_FILES = (
    "tfidf/metadata.json",
    "tfidf/pipeline.joblib",
    "neural/metadata.json",
    "neural/network.pt",
    "neural/vectors.npz",
    "bert/config.json",
    "bert/metadata.json",
    "bert/tokenizer.json",
    "bert/tokenizer_config.json",
    "int8/config.json",
    "int8/metadata.json",
    "int8/tokenizer.json",
    "int8/tokenizer_config.json",
)


def download_verified(url, dest, checksum):
    if dest.is_file() and digest(dest) == checksum:
        return
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        part = Path(tmp.name)
    try:
        with urlopen(url, timeout=120) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if digest(part) != checksum:
            raise ValueError(f"Checksum mismatch for {dest.name}. Retry the download.")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def ensure_training_files(patch):
    required = [FINAL / "artifacts" / name for name in ARTIFACT_FILES] + [patch]
    if all(p.is_file() and p.stat().st_size for p in required):
        return
    print(
        "Downloading the verified training patch and fitted classifiers (48 MB)…",
        flush=True,
    )
    with tempfile.TemporaryDirectory(dir=ROOT / ".work") as tmp:
        cache = Path(tmp)
        bundle = cache / "companion.zip"
        download_verified(COMPANION_URL, bundle, COMPANION_SHA256)
        with zipfile.ZipFile(bundle) as archive:
            for path in (
                *required,
                ROOT / "weights/NOTICE.txt",
                ROOT / "weights/APACHE-2.0.txt",
            ):
                relative = path.relative_to(ROOT)
                archive.extract(relative.as_posix(), cache)
                path.parent.mkdir(parents=True, exist_ok=True)
                (cache / relative).replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-file", type=Path, help="Use the pinned base checkpoint already on disk."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--pack", action="store_true", help="Create the patch from a trained run."
    )
    mode.add_argument(
        "--if-needed",
        action="store_true",
        help="Keep a complete local model installation.",
    )
    args = p.parse_args()
    work = ROOT / ".work"
    work.mkdir(exist_ok=True)
    with FileLock(str(work / "model-setup.lock")):
        folder = FINAL / "artifacts"
        needed = [
            folder / name
            for name in (*ARTIFACT_FILES, "bert/model.safetensors", "int8/int8.pt")
        ]
        if args.if_needed and all(
            path.is_file() and path.stat().st_size for path in needed
        ):
            return
        patch = ROOT / "weights/final-bert.patch.zip"
        if not args.pack:
            ensure_training_files(patch)
        base = args.base_file
        if base is None:
            base = work / "base-model.safetensors"
            download_verified(
                f"https://huggingface.co/{BASE}/resolve/{REVISION}/model.safetensors",
                base,
                BASE_SHA256,
            )
        target = folder / "bert/model.safetensors"
        if args.pack:
            patch.parent.mkdir(exist_ok=True)
            pack(base, target, patch)
            return
        seed()
        restore(base, patch, target)
        export_int8(FINAL)
        subprocess.run(
            [sys.executable, "-m", "scripts.check_int8"], cwd=ROOT, check=True
        )
        print(
            "Restored and verified every FP32 tensor and reloaded the INT8 classifier."
        )


if __name__ == "__main__":
    main()
