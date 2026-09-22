"""Coordinate-safe affine conversions for the VTMOT registration branch.

VTMOT stores a fixed-to-moving homography in ``[x,y,1]`` coordinates, while
the global matcher fits a row-vector affine map in normalised ``[y,x,1]``
feature-grid coordinates.  Keeping this conversion in one tested module avoids
silently supervising the wrong flow convention.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _batched_homography(homography: torch.Tensor) -> torch.Tensor:
    if homography.ndim == 2:
        homography = homography.unsqueeze(0)
    if homography.ndim != 3 or homography.shape[-2:] != (3, 3):
        raise ValueError("homography must be [B,3,3] or [3,3]")
    scale = homography[:, 2:3, 2:3]
    if torch.any(scale.abs() < 1e-8):
        raise ValueError("homography bottom-right value must be non-zero")
    return homography / scale


def _matrix(values: list[list[float]], reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def _normalised_to_feature(image_hw: Tuple[int, int], feature_hw: Tuple[int, int],
                           reference: torch.Tensor) -> torch.Tensor:
    """Map normalised feature ``[y,x,1]`` columns to feature-grid pixels."""
    del image_hw  # Kept in the signature so both conversion paths read alike.
    feature_h, feature_w = map(int, feature_hw)
    if min(feature_h, feature_w) < 2:
        raise ValueError("feature_hw must be at least 2 in both dimensions")
    return _matrix([[(feature_h - 1) / 2, 0.0, (feature_h - 1) / 2],
                    [0.0, (feature_w - 1) / 2, (feature_w - 1) / 2],
                    [0.0, 0.0, 1.0]], reference)


def _feature_to_image(image_hw: Tuple[int, int], feature_hw: Tuple[int, int],
                      reference: torch.Tensor) -> torch.Tensor:
    """Map matcher feature pixels to the image coordinates used by its flow lift."""
    image_h, image_w = map(int, image_hw)
    feature_h, feature_w = map(int, feature_hw)
    if min(image_h, image_w, feature_h, feature_w) < 1:
        raise ValueError("image_hw and feature_hw must be positive")
    return _matrix([[image_h / feature_h, 0.0, 0.0],
                    [0.0, image_w / feature_w, 0.0],
                    [0.0, 0.0, 1.0]], reference)


def homography_to_normalised_affine_yx(homography_xy: torch.Tensor,
                                        image_hw: Tuple[int, int],
                                        feature_hw: Tuple[int, int]) -> torch.Tensor:
    """Convert VTMOT ``[x,y]`` affine GT to matcher ``[B,3,2]`` parameters.

    The return value has the same convention as ``GlobalMatchOutput.affine_yx``:
    ``[y_norm, x_norm, 1] @ affine_yx = [mapped_y_norm, mapped_x_norm]``.
    VTMOT currently contains affine transforms only; failing loudly for a
    projective matrix is safer than silently dropping its perspective terms.
    """
    homography_xy = _batched_homography(homography_xy)
    reference = homography_xy
    if torch.any(homography_xy[:, 2, :2].abs() > 1e-5):
        raise ValueError("affine parameter supervision requires an affine gt_h")
    swap = _matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], reference)
    h_yx = swap.unsqueeze(0) @ homography_xy @ swap.unsqueeze(0)
    normal_to_feature = _normalised_to_feature(image_hw, feature_hw, reference)
    feature_to_image = _feature_to_image(image_hw, feature_hw, reference)
    image_to_feature = torch.linalg.inv(feature_to_image)
    feature_to_normal = torch.linalg.inv(normal_to_feature)
    normal_h_yx = (feature_to_normal.unsqueeze(0) @ image_to_feature.unsqueeze(0)
                   @ h_yx @ feature_to_image.unsqueeze(0)
                   @ normal_to_feature.unsqueeze(0))
    # The matcher stores the transpose because it multiplies row vectors.
    return normal_h_yx.transpose(1, 2)[..., :2]


def normalised_affine_yx_to_homography(affine_yx: torch.Tensor,
                                        image_hw: Tuple[int, int],
                                        feature_hw: Tuple[int, int]) -> torch.Tensor:
    """Convert matcher row-vector parameters back to VTMOT ``[x,y,1]`` GT form."""
    if affine_yx.ndim != 3 or affine_yx.shape[-2:] != (3, 2):
        raise ValueError("affine_yx must be [B,3,2]")
    batch = affine_yx.shape[0]
    normal_h_yx = torch.zeros(batch, 3, 3, dtype=affine_yx.dtype,
                              device=affine_yx.device)
    normal_h_yx[:, :2, :] = affine_yx.transpose(1, 2)
    normal_h_yx[:, 2, 2] = 1.0
    normal_to_feature = _normalised_to_feature(image_hw, feature_hw, affine_yx)
    feature_to_image = _feature_to_image(image_hw, feature_hw, affine_yx)
    h_yx = (feature_to_image.unsqueeze(0) @ normal_to_feature.unsqueeze(0)
            @ normal_h_yx @ torch.linalg.inv(normal_to_feature).unsqueeze(0)
            @ torch.linalg.inv(feature_to_image).unsqueeze(0))
    swap = _matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], affine_yx)
    return swap.unsqueeze(0) @ h_yx @ swap.unsqueeze(0)


def affine_corner_errors(affine_yx: torch.Tensor, homography_xy: torch.Tensor,
                         image_hw: Tuple[int, int], feature_hw: Tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted-vs-GT and GT-inverse cycle errors at image corners.

    Values are pixels.  The cycle diagnostic detects a badly conditioned affine
    estimate in an interpretable coordinate system without introducing another
    training loss.
    """
    predicted = normalised_affine_yx_to_homography(
        affine_yx.float(), image_hw, feature_hw)
    target = _batched_homography(homography_xy.float())
    if target.shape[0] != predicted.shape[0]:
        raise ValueError("affine_yx and homography batch sizes must agree")
    height, width = map(int, image_hw)
    corners = torch.tensor([[0.0, width - 1.0, 0.0, width - 1.0],
                            [0.0, 0.0, height - 1.0, height - 1.0],
                            [1.0, 1.0, 1.0, 1.0]],
                           dtype=predicted.dtype, device=predicted.device)

    def project(matrix: torch.Tensor) -> torch.Tensor:
        mapped = matrix @ corners.unsqueeze(0)
        return mapped[:, :2] / mapped[:, 2:3].clamp_min(1e-8)

    target_points, predicted_points = project(target), project(predicted)
    corner_epe = torch.linalg.vector_norm(predicted_points - target_points, dim=1).mean(dim=1)
    cycle_points = project(predicted @ torch.linalg.inv(target))
    cycle_epe = torch.linalg.vector_norm(cycle_points - corners[:2].unsqueeze(0), dim=1).mean(dim=1)
    return corner_epe, cycle_epe
