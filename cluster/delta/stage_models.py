#!/usr/bin/env python3
"""Stage and validate the LongNav adapter and its base model for offline jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = "Aasdfip/hm3d_rpp_ke_standard-checkpoint_231"


def parse_args() -> argparse.Namespace:
    runtime_root = Path(
        os.environ.get(
            "LONGNAV_RUNTIME_ROOT",
            "/work/nvme/bgon/brabiei/longnav_runtime",
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-revision", default="main")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--base-revision", default="main")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("LONGNAV_MODEL_ROOT", runtime_root / "models")),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            os.environ.get(
                "LONGNAV_MODEL_MANIFEST",
                runtime_root / "models" / "manifest.json",
            )
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-path", choices=("adapter", "base"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def repo_slug(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id)


def file_inventory(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
            }
        )
    return files


def validate_weight_index(model_dir: Path) -> None:
    for index_path in model_dir.glob("*.index.json"):
        weight_map = read_json(index_path).get("weight_map", {})
        missing = sorted(
            {filename for filename in weight_map.values() if not (model_dir / filename).is_file()}
        )
        if missing:
            raise RuntimeError(
                f"{index_path.name} references missing weight files: {', '.join(missing)}"
            )


def validate_safetensors(model_dir: Path) -> None:
    from safetensors import safe_open

    for weights_path in model_dir.glob("*.safetensors"):
        with safe_open(weights_path, framework="pt", device="cpu") as weights:
            if not list(weights.keys()):
                raise RuntimeError(f"No tensors found in {weights_path}")


def validate_manifest(manifest: dict[str, Any]) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    adapter_dir = Path(manifest["checkpoint"]["path"])
    base_dir = Path(manifest["base_model"]["path"])
    adapter_config = adapter_dir / "adapter_config.json"

    if not adapter_config.is_file():
        raise FileNotFoundError(f"Missing adapter configuration: {adapter_config}")
    if Path(read_json(adapter_config).get("base_model_name_or_path", "")) != base_dir:
        raise RuntimeError("The staged adapter does not point to the staged base model")

    adapter_weights = list(adapter_dir.glob("*.safetensors")) + list(
        adapter_dir.glob("*.bin")
    )
    base_weights = list(base_dir.glob("*.safetensors")) + list(
        base_dir.glob("pytorch_model*.bin")
    )
    if not adapter_weights:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")
    if not base_weights:
        raise FileNotFoundError(f"No base-model weights found in {base_dir}")
    validate_weight_index(base_dir)
    validate_safetensors(adapter_dir)
    validate_safetensors(base_dir)

    from peft import PeftConfig
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    AutoConfig.from_pretrained(base_dir, local_files_only=True, trust_remote_code=True)
    AutoTokenizer.from_pretrained(
        base_dir,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    AutoProcessor.from_pretrained(
        base_dir,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    PeftConfig.from_pretrained(adapter_dir, local_files_only=True)

    print(f"Offline model validation: OK\n  adapter: {adapter_dir}\n  base:    {base_dir}")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model manifest not found: {path}\n"
            "Run cluster/delta/stage_models.sh on a login node first."
        )
    return read_json(path)


def stage_models(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is unavailable; run cluster/delta/setup.sh first"
        ) from error

    api = HfApi()
    try:
        checkpoint_sha = api.model_info(
            args.checkpoint, revision=args.checkpoint_revision
        ).sha
    except Exception as error:
        raise RuntimeError(
            f"Cannot access Hugging Face checkpoint {args.checkpoint!r}. "
            "If it is gated, run `hf auth login` on the login node."
        ) from error

    checkpoint_dir = (
        args.model_root / "checkpoints" / repo_slug(args.checkpoint) / checkpoint_sha
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.checkpoint,
        revision=checkpoint_sha,
        local_dir=checkpoint_dir,
        allow_patterns=("*.json", "*.bin", "*.safetensors", "*.pt", "*.yaml"),
        ignore_patterns=("*.msgpack", "*.h5"),
    )

    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint is not a PEFT adapter; missing {adapter_config_path}"
        )
    upstream_config_path = checkpoint_dir / "adapter_config.upstream.json"
    if upstream_config_path.is_file():
        upstream_config = read_json(upstream_config_path)
    else:
        upstream_config = read_json(adapter_config_path)
        write_json_atomic(upstream_config_path, upstream_config)

    adapter_config = dict(upstream_config)
    upstream_base = upstream_config.get("base_model_name_or_path")
    base_repo_id = args.base_model or upstream_base
    if (
        not isinstance(base_repo_id, str)
        or Path(base_repo_id).is_absolute()
        or "/" not in base_repo_id
    ):
        raise ValueError(
            "Could not determine the base Hugging Face repository. "
            "Pass it explicitly with --base-model."
        )

    try:
        base_sha = api.model_info(base_repo_id, revision=args.base_revision).sha
    except Exception as error:
        raise RuntimeError(
            f"Cannot access Hugging Face base model {base_repo_id!r}. "
            "If it is gated, run `hf auth login` on the login node."
        ) from error

    base_dir = args.model_root / "base" / repo_slug(base_repo_id) / base_sha
    base_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=base_repo_id,
        revision=base_sha,
        local_dir=base_dir,
        ignore_patterns=("*.msgpack", "*.h5", "*.onnx", "tf_model.*"),
    )

    adapter_config["base_model_name_or_path"] = str(base_dir.resolve())
    write_json_atomic(adapter_config_path, adapter_config)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "repo_id": args.checkpoint,
            "requested_revision": args.checkpoint_revision,
            "resolved_revision": checkpoint_sha,
            "path": str(checkpoint_dir.resolve()),
            "files": file_inventory(checkpoint_dir),
        },
        "base_model": {
            "repo_id": base_repo_id,
            "upstream_reference": upstream_base,
            "requested_revision": args.base_revision,
            "resolved_revision": base_sha,
            "path": str(base_dir.resolve()),
            "files": file_inventory(base_dir),
        },
    }
    validate_manifest(manifest)
    write_json_atomic(args.manifest, manifest)
    print(f"Wrote model manifest: {args.manifest}")
    return manifest


def main() -> int:
    args = parse_args()
    try:
        if args.print_path:
            manifest = load_manifest(args.manifest)
            key = "checkpoint" if args.print_path == "adapter" else "base_model"
            print(manifest[key]["path"])
            return 0
        if args.validate_only:
            validate_manifest(load_manifest(args.manifest))
            return 0
        stage_models(args)
        return 0
    except Exception as error:
        print(f"Model staging failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
