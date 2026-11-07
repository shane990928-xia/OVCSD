"""Compare the post-update first-layer norm directly with the SFT on CPU.

Run with torchrun and the same world size used to save the FSDP checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor", type=Path)
    parser.add_argument("sft", type=Path)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, size = dist.get_rank(), dist.get_world_size()
    key = "model.layers.0.input_layernorm.weight"
    idx = json.loads((args.sft / "model.safetensors.index.json").read_text())
    with safe_open(args.sft / idx["weight_map"][key], framework="pt") as source:
        original = source.get_tensor(key).float()
    path = args.actor / f"model_world_size_{size}_rank_{rank}.pt"
    # Load only checkpoints produced locally by this trainer.
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    param = state[key]
    if isinstance(param, DTensor):
        mesh, placements = param.device_mesh, param.placements
        _, offsets = compute_local_shape_and_global_offset(param.shape, mesh, placements)
        coord = mesh.get_coordinate()
        primary = all(not p.is_replicate() or c == 0 for p, c in zip(placements, coord))
        chunks = [(offsets[0], param.to_local())] if primary else []
    elif hasattr(param, "local_shards"):
        chunks = [(shard.metadata.shard_offsets[0], shard.tensor) for shard in param.local_shards()]
    else:
        chunks = [(0, param)] if rank == 0 else []
    absolute = torch.zeros(1)
    changed = torch.zeros(1, dtype=torch.int64)
    for start, value in chunks:
        diff = (value.float() - original[start:start + value.numel()]).abs()
        absolute = torch.maximum(absolute, diff.max())
        changed += (diff != 0).sum()
    dist.all_reduce(absolute, op=dist.ReduceOp.MAX)
    dist.all_reduce(changed)
    if not absolute.item() > 0 or not torch.isfinite(absolute).all():
        raise RuntimeError("No finite parameter change detected")
    opt = torch.load(args.actor / f"optim_world_size_{size}_rank_{rank}.pt",
                     map_location="cpu", weights_only=False, mmap=True)
    steps = [int(slot["step"]) for slot in opt["state"].values() if "step" in slot]
    if not steps or min(steps) < 1:
        raise RuntimeError("Missing AdamW update in optimizer shard")
    minimum = torch.tensor(min(steps))
    maximum = torch.tensor(max(steps))
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(json.dumps({"parameter": key, "numel": original.numel(),
                          "changed_elements": changed.item(), "max_abs_delta": absolute.item(),
                          "adam_step_min": minimum.item(), "adam_step_max": maximum.item()}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
