"""List run checkpoints, and work out which warm start produced one.

A checkpoint whose coarse stage was frozen shares those tensors bit for bit
with whatever ``--init`` it was trained from, because freezing means they were
never updated.  Hashing exactly the submodules that ``freeze_coarse`` holds
fixed therefore identifies the *coarse lineage* from the artefacts alone --
note that a whole chain shares it, so this alone does not name the immediate
parent.  For that, the trainer's own metrics are used: the step-0 validation of
a run is the validation of its warm-start weights, so the parent's final
``val_epe_px`` equals it exactly.  Checkpoints written since provenance
recording exists also carry their own command line, which is reported too.

    python -m res.checkpoint_provenance --root res_runs
    python -m res.checkpoint_provenance --root res_runs --init-of res_runs/structural_iterative_r2/last.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

# Held fixed by freeze_coarse_parameters: everything except the 1/4 stage.
TRAINABLE_PREFIXES = ("fine_interaction.", "fine_cross_attention.",
                      "iterative_refinement.", "local_matcher.",
                      "ir_feature_adapter.")


def _metrics(path: Path) -> list[dict]:
    """The per-step records a run wrote next to its checkpoints."""
    metrics = Path(path).parent / "metrics.jsonl"
    if not metrics.is_file():
        return []
    rows = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def warm_start_epe(path: Path) -> float | None:
    """The validation EPE this run recorded for the weights it started from."""
    rows = _metrics(path)
    return rows[0].get("val_epe_px") if rows else None


def final_epe(path: Path) -> float | None:
    """The validation EPE of this checkpoint's own last recorded validation."""
    rows = _metrics(path)
    return rows[-1].get("val_epe_px") if rows else None


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
    """Candidates sharing the reference's frozen stage, the warm start first.

    A whole chain inherits one frozen coarse stage, so the digest narrows the
    field to that chain.  Within it, the run whose *final* validation equals the
    reference's *step-0* validation is the immediate parent: those are the same
    weights scored on the same frames.
    """
    target = describe(reference)
    started_from = warm_start_epe(reference)
    target["warm_start_epe"] = started_from
    rows = []
    for path in candidates(root, reference):
        row = describe(path)
        row["final_epe"] = final_epe(path)
        row["is_warm_start"] = (
            started_from is not None and row["final_epe"] is not None
            and abs(row["final_epe"] - started_from) < 1e-6)
        rows.append(row)
    matches = [row for row in rows if row["digest"] == target["digest"]]
    matches.sort(key=lambda row: (not row["is_warm_start"], row["loop"],
                                 row["step"] or 0))
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
        print(f"  its step-0 validation (the warm-start weights): "
              f"val_epe_px={target['warm_start_epe']}")
        if not matches:
            print("  no checkpoint here shares its frozen coarse stage, so the "
                  "warm start came from outside --root")
        for row in matches:
            mark = "  <- WARM START" if row["is_warm_start"] else ""
            print(f"  {row['path']} step={row['step']} loop={row['loop']} "
                  f"xattn={row['cross_attention']} "
                  f"final_val_epe={row['final_epe']} "
                  f"recorded_init={row['init']}{mark}")
        return
    for path in candidates(args.root):
        row = describe(path)
        print(f"{row['path']:58s} step={str(row['step']):>5s} "
              f"loop={int(row['loop'])} xattn={int(row['cross_attention'])} "
              f"digest={row['digest']} init={row['init']}")


if __name__ == "__main__":
    main()
