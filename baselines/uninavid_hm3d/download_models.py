"""Download and verify the pinned Uni-NaVid and EVA-ViT-G checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import requests
from huggingface_hub import snapshot_download


MODEL_REPO = "Jzzhang/Uni-NaVid"
MODEL_REVISION = "0437222534b2a19abeb20710bfd53c44c5bf3dd4"
MODEL_DIRNAME = "uninavid-7b-full-224-video-fps-1-grid-2"
EVA_URL = (
    "https://storage.googleapis.com/sfr-vision-language-research/"
    "LAVIS/models/BLIP2/eva_vit_g.pth"
)
EVA_SIZE = 2_025_249_237
EVA_MD5 = "c483374fc17b72dc61e016680345a0ff"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_eva(destination: Path) -> None:
    if destination.exists():
        if destination.stat().st_size == EVA_SIZE and file_md5(destination) == EVA_MD5:
            print(f"EVA encoder already verified: {destination}")
            return
        raise RuntimeError(f"Existing EVA checkpoint failed verification: {destination}")

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    mode = "ab" if offset else "wb"

    with requests.get(EVA_URL, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
            mode = "wb"
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)

    if partial.stat().st_size != EVA_SIZE:
        raise RuntimeError(
            f"EVA checkpoint has {partial.stat().st_size} bytes; expected {EVA_SIZE}"
        )
    checksum = file_md5(partial)
    if checksum != EVA_MD5:
        raise RuntimeError(f"EVA checkpoint MD5 {checksum}; expected {EVA_MD5}")
    os.replace(partial, destination)
    print(f"Downloaded and verified EVA encoder: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1] / "Uni-NaVid"
    parser.add_argument("--uninavid-root", type=Path, default=default_root)
    args = parser.parse_args()

    model_zoo = args.uninavid_root.resolve() / "model_zoo"
    model_zoo.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=[f"{MODEL_DIRNAME}/*"],
        local_dir=model_zoo,
    )

    model_dir = model_zoo / MODEL_DIRNAME
    required = [
        model_dir / "config.json",
        model_dir / "pytorch_model.bin.index.json",
        model_dir / "pytorch_model-00001-of-00002.bin",
        model_dir / "pytorch_model-00002-of-00002.bin",
        model_dir / "tokenizer.model",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Model snapshot is incomplete: {missing}")

    download_eva(model_zoo / "eva_vit_g.pth")
    print(f"Uni-NaVid model verified: {model_dir}")


if __name__ == "__main__":
    main()

