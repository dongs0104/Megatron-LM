#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Extract a smaller sub-model from a Matformer checkpoint.

A Matformer model is trained with nested FFN granularities.  After training,
this script extracts a sub-model that uses only the first ``--target-ffn-size``
neurons of every FFN hidden dimension, producing a checkpoint that can be loaded
by a *standard* GPT model whose ``ffn_hidden_size`` equals ``--target-ffn-size``.

Usage example::

    python tools/matformer_extract_submodel.py \\
        --input-checkpoint /path/to/matformer_iter_100000 \\
        --output-checkpoint /path/to/submodel_iter_100000 \\
        --target-ffn-size 2048

The script handles the typical ColumnParallelLinear / RowParallelLinear weight
layout used by Megatron-LM tensor parallelism.  It does *not* require a running
GPU cluster—weights are sliced on CPU.

Checkpoint format assumptions
------------------------------
* Megatron-LM distributed checkpoint (one ``model_optim_rng.pt`` per TP/PP rank,
  produced by ``--save-interval`` during pre-training).
* Tensor-parallel sharding of ``linear_fc1.weight`` along axis 0 (column
  parallel) and ``linear_fc2.weight`` along axis 1 (row parallel).
* Gated linear units (SwiGLU / GEGLU) double the ``linear_fc1`` output dim, so
  the actual weight shape is ``[2 * ffn_hidden_size / tp, hidden_size]``.  The
  script detects this automatically.
"""

import argparse
import os
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Weight slicing helpers
# ---------------------------------------------------------------------------

def _slice_fc1_weight(weight: torch.Tensor, target_local: int, is_gated: bool) -> torch.Tensor:
    """Slice a ColumnParallelLinear fc1 weight tensor.

    Args:
        weight: Shape ``[full_local_out, in_features]`` or
                ``[2 * full_local_out, in_features]`` for gated units.
        target_local: Number of output neurons to keep *per TP rank*.
        is_gated: Whether this layer uses a gated linear unit (e.g. SwiGLU).

    Returns:
        Sliced weight of shape ``[target_local, in_features]`` or
        ``[2 * target_local, in_features]`` for gated units.
    """
    if is_gated:
        full_local = weight.shape[0] // 2
        gate = weight[:full_local][:target_local]
        up = weight[full_local:][:target_local]
        return torch.cat([gate, up], dim=0)
    else:
        return weight[:target_local]


def _slice_fc1_bias(bias: torch.Tensor, target_local: int, is_gated: bool) -> torch.Tensor:
    if is_gated:
        full_local = bias.shape[0] // 2
        return torch.cat([bias[:full_local][:target_local], bias[full_local:][:target_local]])
    else:
        return bias[:target_local]


def _slice_fc2_weight(weight: torch.Tensor, target_local: int) -> torch.Tensor:
    """Slice a RowParallelLinear fc2 weight tensor.

    Args:
        weight: Shape ``[out_features, full_local_in]``.
        target_local: Number of input neurons to keep *per TP rank*.

    Returns:
        Sliced weight of shape ``[out_features, target_local]``.
    """
    return weight[:, :target_local]


# ---------------------------------------------------------------------------
# Checkpoint processing
# ---------------------------------------------------------------------------

def _process_state_dict(
    state_dict: dict,
    full_ffn_size: int,
    target_ffn_size: int,
    tp_size: int,
) -> dict:
    """Slice all Matformer MLP weights in *state_dict* to *target_ffn_size*.

    The function operates on the flat key-value pairs of a model state dict.
    Keys are expected to follow the Megatron-LM naming convention, e.g.::

        decoder.layers.0.mlp.linear_fc1.weight
        decoder.layers.0.mlp.linear_fc2.weight

    Args:
        state_dict: Flat ``{str: Tensor}`` mapping from a single TP-rank file.
        full_ffn_size: The FFN hidden size the checkpoint was trained with.
        target_ffn_size: The desired sub-model FFN hidden size.
        tp_size: Tensor-parallel world size.

    Returns:
        A new state dict with sliced MLP weights.
    """
    full_local = full_ffn_size // tp_size
    target_local = target_ffn_size // tp_size

    new_sd = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            new_sd[key] = value
            continue

        if "mlp.linear_fc1.weight" in key:
            is_gated = value.shape[0] in (2 * full_local,)
            new_sd[key] = _slice_fc1_weight(value, target_local, is_gated)
        elif "mlp.linear_fc1.bias" in key:
            is_gated = value.shape[0] in (2 * full_local,)
            new_sd[key] = _slice_fc1_bias(value, target_local, is_gated)
        elif "mlp.linear_fc2.weight" in key:
            new_sd[key] = _slice_fc2_weight(value, target_local)
        elif "mlp.linear_fc2.bias" in key:
            # fc2 bias is replicated (shape [hidden_size]) — keep as is.
            new_sd[key] = value
        else:
            new_sd[key] = value

    return new_sd


def _load_checkpoint_file(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_checkpoint_file(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-checkpoint",
        required=True,
        type=Path,
        help="Path to the input Matformer checkpoint directory (e.g. iter_0100000/).",
    )
    parser.add_argument(
        "--output-checkpoint",
        required=True,
        type=Path,
        help="Path for the output sub-model checkpoint directory.",
    )
    parser.add_argument(
        "--target-ffn-size",
        required=True,
        type=int,
        help="FFN hidden size of the extracted sub-model.  Must be one of the granularities "
             "used during Matformer training.",
    )
    parser.add_argument(
        "--full-ffn-size",
        type=int,
        default=None,
        help="FFN hidden size of the Matformer model.  Auto-detected from checkpoint if "
             "not specified.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor-parallel world size the checkpoint was saved with.  Default: 1.",
    )
    args = parser.parse_args()

    in_dir: Path = args.input_checkpoint
    out_dir: Path = args.output_checkpoint
    target: int = args.target_ffn_size
    tp_size: int = args.tp_size

    if not in_dir.exists():
        print(f"ERROR: input checkpoint directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all .pt shard files.
    shard_files = sorted(in_dir.rglob("*.pt"))
    if not shard_files:
        print(f"ERROR: no .pt files found under {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_files)} shard file(s) in {in_dir}")

    # Auto-detect full FFN size from first model shard if not provided.
    full_ffn_size = args.full_ffn_size
    if full_ffn_size is None:
        first_data = _load_checkpoint_file(shard_files[0])
        model_sd = first_data.get("model", first_data)
        for key, val in model_sd.items():
            if "mlp.linear_fc2.weight" in key and isinstance(val, torch.Tensor):
                # fc2 weight: [out_features, local_in] → local_in * tp = full_ffn_size
                full_ffn_size = val.shape[1] * tp_size
                print(f"Auto-detected full_ffn_size={full_ffn_size} from key '{key}'")
                break
        if full_ffn_size is None:
            print(
                "ERROR: could not auto-detect full_ffn_size.  "
                "Please pass --full-ffn-size explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

    if target > full_ffn_size:
        print(
            f"ERROR: --target-ffn-size {target} exceeds full ffn size {full_ffn_size}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if target % tp_size != 0:
        print(
            f"ERROR: --target-ffn-size {target} is not divisible by --tp-size {tp_size}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Extracting sub-model: ffn_hidden_size {full_ffn_size} -> {target} "
        f"(tp_size={tp_size})"
    )

    # Process each shard.
    for shard_path in shard_files:
        rel = shard_path.relative_to(in_dir)
        out_path = out_dir / rel
        print(f"  {rel} ... ", end="", flush=True)

        data = _load_checkpoint_file(shard_path)

        # Handle both bare state-dicts and Megatron checkpoint dicts.
        if "model" in data:
            data["model"] = _process_state_dict(
                data["model"], full_ffn_size, target, tp_size
            )
        else:
            data = _process_state_dict(data, full_ffn_size, target, tp_size)

        _save_checkpoint_file(data, out_path)
        print("done")

    print(f"\nSub-model checkpoint written to: {out_dir}")
    print(
        f"To load it, create a TransformerConfig with ffn_hidden_size={target} "
        f"(no matformer_ffn_granularities needed for inference)."
    )


if __name__ == "__main__":
    main()
