from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_pinball_loss(
    y_true: torch.Tensor,
    y_pred_q: torch.Tensor,
    q: float = 0.9,
) -> torch.Tensor:
    """
    y_true, y_pred_q: [B, H]
    """
    diff = y_true - y_pred_q
    return torch.maximum(q * diff, (q - 1.0) * diff).mean()


def masked_mae(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    y_true, y_pred, mask: [B, H]
    mask should be 0/1 or bool.
    """
    mask = mask.float()
    abs_err = torch.abs(y_true - y_pred)
    return (abs_err * mask).sum() / (mask.sum() + eps)


def huber_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Smooth alternative to MAE for point forecast training.
    """
    return F.huber_loss(y_pred, y_true, delta=delta, reduction="mean")


def competition_aligned_loss(
    y_true: torch.Tensor,
    point_pred: torch.Tensor,
    p90_pred: torch.Tensor,
    peak_threshold: float | torch.Tensor,
    point_loss_type: str = "huber",
    point_huber_delta: float = 1.0,
    weight_mae_all: float = 0.5,
    weight_mae_peak: float = 0.3,
    weight_p90: float = 0.2,
    monotonicity_weight: float = 0.05,
) -> torch.Tensor:
    """
    Approximate the competition metric:
      0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90

    Inputs are normalized-space tensors if training on normalized targets.
    That's okay as long as threshold is also in normalized space.
    """
    if point_loss_type == "mae":
        loss_all = torch.abs(y_true - point_pred).mean()
    elif point_loss_type == "huber":
        loss_all = huber_loss(y_true, point_pred, delta=point_huber_delta)
    else:
        raise ValueError(f"Unsupported point_loss_type: {point_loss_type}")

    if not torch.is_tensor(peak_threshold):
        peak_threshold = torch.tensor(peak_threshold, device=y_true.device, dtype=y_true.dtype)

    peak_mask = y_true >= peak_threshold

    if peak_mask.any():
        loss_peak = masked_mae(y_true, point_pred, peak_mask)
    else:
        loss_peak = torch.abs(y_true - point_pred).mean()

    loss_p90 = torch_pinball_loss(y_true, p90_pred, q=0.9)

    # Penalize invalid quantile ordering: p90 should be >= point
    monotonicity_penalty = F.relu(point_pred - p90_pred).mean()

    total = (
        weight_mae_all * loss_all
        + weight_mae_peak * loss_peak
        + weight_p90 * loss_p90
        + monotonicity_weight * monotonicity_penalty
    )
    return total


def point_only_peak_weighted_loss(
    y_true: torch.Tensor,
    point_pred: torch.Tensor,
    peak_threshold: float | torch.Tensor,
    base_loss: str = "huber",
    huber_delta: float = 1.0,
    weight_all: float = 0.7,
    weight_peak: float = 0.3,
) -> torch.Tensor:
    """
    Stage-1 point model training:
    emphasize strong point forecast, with extra focus on peaks.
    """
    if base_loss == "mae":
        loss_all = torch.abs(y_true - point_pred).mean()
    elif base_loss == "huber":
        loss_all = huber_loss(y_true, point_pred, delta=huber_delta)
    else:
        raise ValueError(f"Unsupported base_loss: {base_loss}")

    if not torch.is_tensor(peak_threshold):
        peak_threshold = torch.tensor(peak_threshold, device=y_true.device, dtype=y_true.dtype)

    peak_mask = y_true >= peak_threshold
    if peak_mask.any():
        loss_peak = masked_mae(y_true, point_pred, peak_mask)
    else:
        loss_peak = torch.abs(y_true - point_pred).mean()

    return weight_all * loss_all + weight_peak * loss_peak