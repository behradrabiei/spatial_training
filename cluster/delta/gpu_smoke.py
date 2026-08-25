#!/usr/bin/env python3
"""Validate the CUDA features used by LongNav."""

import sys

import torch
import torch.nn.functional as functional


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    supported = {(8, 0), (8, 6), (9, 0)}
    if capability not in supported:
        print(f"Unexpected GPU compute capability: {capability}", file=sys.stderr)
        return 1
    if not torch.cuda.is_bf16_supported():
        print("This GPU does not support BF16", file=sys.stderr)
        return 1

    query = torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16)
    output = functional.scaled_dot_product_attention(query, query, query)
    torch.cuda.synchronize()
    if not torch.isfinite(output).all():
        print("SDPA returned non-finite values", file=sys.stderr)
        return 1

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"Compute capability: {capability}; BF16/SDPA: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
