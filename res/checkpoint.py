"""Checkpoint loading that tolerates newly added, still-untrained submodules.

An architecture may gain a stage after a checkpoint was written -- the
structural-prior route gained its 1/4 local matcher this way.  ``strict=True``
then refuses to load even though every stored tensor still applies, while a bare
``strict=False`` would also hide genuinely broken checkpoints.  These helpers
allow exactly the module prefixes that are expected to be new and treat anything
else as an error.

Trainer ``--init`` stays deliberately more lenient than this (it also skips
tensors whose *shape* changed, because width is a tuning knob there).  An
evaluation must not do that: silently dropping a resized tensor would score a
half-random model.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch

# Submodules that may legitimately be absent from a checkpoint because they were
# introduced afterwards.  A checkpoint is allowed to be missing these and nothing
# else.
OPTIONAL_MODULE_PREFIXES: tuple[str, ...] = (
    "local_matcher.", "coarse_refiner.", "fine_interaction.",
    "fine_cross_attention.", "ir_feature_adapter.")


def split_state_dict(state_dict: Mapping[str, torch.Tensor], model: torch.nn.Module
                     ) -> tuple[dict, list[str]]:
    """Keep the tensors whose name and shape still match ``model``."""
    current = model.state_dict()
    keep: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for name, value in state_dict.items():
        if name in current and current[name].shape == value.shape:
            keep[name] = value
        else:
            dropped.append(name)
    return keep, sorted(dropped)


def load_registration_state(model: torch.nn.Module,
                            state_dict: Mapping[str, torch.Tensor], *,
                            optional_prefixes: Iterable[str] = OPTIONAL_MODULE_PREFIXES
                            ) -> dict:
    """Load a checkpoint, allowing only ``optional_prefixes`` to start untrained.

    Returns a report naming the modules that begin untrained, so a caller cannot
    mistake a freshly built stage for a restored one.
    """
    optional = tuple(optional_prefixes)
    current = model.state_dict()
    keep, dropped = split_state_dict(state_dict, model)
    unknown = [name for name in dropped if name not in current]
    if unknown:
        raise ValueError(
            f"model {type(model).__name__} does not define these checkpoint keys: "
            f"{unknown[:5]}")
    if dropped:
        raise ValueError(
            "checkpoint tensors have a different shape than the model; refusing to "
            f"evaluate a partially loaded model: {dropped[:5]}")
    result = model.load_state_dict(keep, strict=False)
    missing_required = [name for name in result.missing_keys
                        if not name.startswith(optional)]
    if missing_required:
        raise ValueError(
            f"checkpoint is missing required tensors: {missing_required[:5]}")
    return {"kept": len(keep), "untrained": sorted(result.missing_keys)}
