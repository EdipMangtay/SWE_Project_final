"""Losslessly restore final FP32 weights from the pinned public base and measured training changes."""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-file",
        type=Path,
        help="Use the pinned public model.safetensors already on disk.",
    )
    p.add_argument(
        "--pack",
        action="store_true",
        help="Create the portable patch from a completed final training run.",
    )
    args = p.parse_args()
    base = args.base_file
    if base is None:
        from huggingface_hub import hf_hub_download

        base = Path(hf_hub_download(BASE, "model.safetensors", revision=REVISION))
    patch = ROOT / "weights" / "final-bert.patch.zip"
    dest = FINAL / "artifacts/bert/model.safetensors"
    if args.pack:
        patch.parent.mkdir(exist_ok=True)
        pack(base, dest, patch)
    else:
        seed()
        restore(base, patch, dest)
        export_int8(FINAL)
        print("Restored and verified every FP32 tensor, then exported the INT8 model.")
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "scripts.check_int8"], cwd=ROOT, check=True
        )


if __name__ == "__main__":
    main()
