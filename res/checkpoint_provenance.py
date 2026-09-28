"""List run checkpoints, and work out which warm start produced one.

A checkpoint whose coarse stage was frozen shares those tensors bit for bit
with whatever ``--init`` it was trained from, because freezing means they were
never updated.  Hashing exactly the submodules that ``freeze_coarse`` holds
fixed therefore identifies the parent from the artefacts alone, without any
recorded metadata.  Checkpoints written since provenance recording exists also
carry their own command line, which is reported when present.

    python -m res.checkpoint_provenance --root res_runs
    python -m res.checkpoint_provenance --root res_runs --init-of res_runs/structural_iterative_r2/last.pt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

# Held fixed by freeze_coarse_parameters: everything except the 1/4 stage.
TRAINABLE_PREFIXES = ("fine_interaction.", "fine_cross_attention.",
                      "iterative_refinement.", "local_matcher.",
                      "ir_feature_adapter.")


def frozen_keys(model: dict) -> list[str]:
    """The tensor names a frozen coarse stage guarantees to be inherited."""
    return sorted(name for name in model
                  if not name.startswith(TRAINABLE_PREFIXES))


def frozen_digest(model: dict, keys: list[str] | None = None) -> str:
    """A short digest of the frozen submodules, independent of tensor order."""
    names = frozen_keys(model) if keys is None else keys
    digest = hashlib.sha256()
    for name in names:
        value = model.get(name)
        if value is None or not torch.is_tensor(value):
            continue
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def describe(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = checkpoint["model"]
    config = checkpoint.get("config") or {}
    return {"path": str(path),
            "step": checkpoint.get("step"),
            "loop": bool((config.get("iterative_refinement") or {}).get("enabled")),
            "cross_attention": bool(
                (config.get("fine_cross_attention") or {}).get("enabled")),
            "init": (checkpoint.get("provenance") or {}).get("init"),
            "digest": frozen_digest(model)}


def candidates(root: Path, reference: Path | None = None) -> list[Path]:
    found = sorted(root.glob("*/last.pt")) + sorted(root.glob("*/best.pt"))
    return [path for path in found
            if reference is None or path.resolve() != reference.resolve()]


def find_init(root: Path, reference: Path) -> tuple[dict, list[dict]]:
    """Every candidate whose frozen stage is identical to the reference's."""
    target = describe(reference)
    rows = [describe(path) for path in candidates(root, reference)]
    matches = [row for row in rows if row["digest"] == target["digest"]]
    # A descendant inherits the same frozen stage, so a match with its loop
    # enabled is a later run, not the parent.
    matches.sort(key=lambda row: (row["loop"], row["step"] or 0))
    return target, matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("res_runs"))
    parser.add_argument("--init-of", type=Path, default=None,
                        help="report which checkpoints share this one's frozen stage")
    args = parser.parse_args()
    if args.init_of is not None:
        target, matches = find_init(args.root, args.init_of)
        print(f"reference {target['path']} step={target['step']} "
              f"loop={target['loop']} digest={target['digest']}")
        parents = [row for row in matches if not row["loop"]]
        descendants = [row for row in matches if row["loop"]]
        for row in parents:
            print(f"  parent? {row['path']} step={row['step']} "
                  f"xattn={row['cross_attention']} recorded_init={row['init']}")
        if not parents:
            print("  no parent found: every match has the loop enabled, so the "
                  "warm start is not among these checkpoints")
        for row in descendants:
            print(f"  shares this frozen stage (later run): {row['path']} "
                  f"step={row['step']}")
        return
    for path in candidates(args.root):
        row = describe(path)
        print(f"{row['path']:58s} step={str(row['step']):>5s} "
              f"loop={int(row['loop'])} xattn={int(row['cross_attention'])} "
              f"digest={row['digest']} init={row['init']}")


if __name__ == "__main__":
    main()
