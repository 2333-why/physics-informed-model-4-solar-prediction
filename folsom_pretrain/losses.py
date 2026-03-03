from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_smoothness_loss(seq: torch.Tensor) -> torch.Tensor:
    """L2 smoothness on adjacent timesteps."""
    if seq.shape[1] <= 1:
        return seq.new_tensor(0.0)
    diff = seq[:, 1:, :] - seq[:, :-1, :]
    return (diff * diff).mean()


def pretrain_loss(
    ghi_pred: torch.Tensor,
    ghi_true: torch.Tensor,
    cloud_params: torch.Tensor,
    lambda_smooth: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon = F.mse_loss(ghi_pred, ghi_true)
    smooth = temporal_smoothness_loss(cloud_params)
    total = recon + lambda_smooth * smooth
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_recon": float(recon.detach().cpu()),
        "loss_smooth": float(smooth.detach().cpu()),
    }
