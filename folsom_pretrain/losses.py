from __future__ import annotations

import torch


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def _masked_mean(v: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = m.to(v.dtype)
    return (v * w).sum() / (w.sum() + eps)


def temporal_l1(seq: torch.Tensor) -> torch.Tensor:
    if seq.shape[1] <= 1:
        return seq.new_tensor(0.0)
    return (seq[:, 1:] - seq[:, :-1]).abs().mean()


def physics_loss(
    out: dict[str, torch.Tensor],
    target_rad: torch.Tensor,
    clear_mask: torch.Tensor,
    stage: int,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    # target_rad: [B,T,3] -> ghi,dni,dhi
    ghi = target_rad[..., 0]
    dni = target_rad[..., 1]
    dhi = target_rad[..., 2]

    s_ghi = float(cfg.get("scale_ghi", 1000.0))
    s_dni = float(cfg.get("scale_dni", 1000.0))
    s_dhi = float(cfg.get("scale_dhi", 1000.0))

    wg = cfg.get("w_g", 0.7)
    wn = cfg.get("w_n", 1.0)
    wd = cfg.get("w_d", 0.7)

    l_rad = (
        wg * charbonnier((out["ghi_hat"] - ghi) / s_ghi).mean()
        + wn * charbonnier((out["dni_hat"] - dni) / s_dni).mean()
        + wd * charbonnier((out["dhi_hat"] - dhi) / s_dhi).mean()
    )

    l_split = charbonnier((out["dhi_cld"] - (out["ghi_cld"] - out["fd"])) / s_dhi).mean()

    mu0 = out.get("mu0")
    if mu0 is None:
        mu0 = torch.clamp((out["ghi_hat"] - out["dhi_hat"]) / out["dni_hat"].clamp_min(1e-3), 0.0, 1.0)
    l_id = charbonnier((out["ghi_hat"] - (out["dni_hat"] * mu0 + out["dhi_hat"])) / s_ghi).mean()

    l_dt_all = torch.relu(out["tdt_clr"] - 1.0)
    l_dt = _masked_mean(l_dt_all, clear_mask) if stage == 1 else l_dt_all.mean()
    l_delta = (out["d_tdu"].abs() + out["d_tuu"].abs() + out["d_ruu"].abs()).mean()
    l_base = out["base_params_l2"] / out["ghi_hat"].numel()

    l_clr = _masked_mean(
        charbonnier((out["ghi_clr_pv"] - ghi) / s_ghi) + charbonnier((out["dni_clr_pv"] - dni) / s_dni),
        clear_mask,
    )
    l_f0 = _masked_mean(charbonnier(out["f"]), clear_mask)

    l_aod = temporal_l1(out["aod"])
    l_cloud_smooth = temporal_l1(out["tau_cld"]) + temporal_l1(out["f"])
    l_proxy = _masked_mean(
        charbonnier((out["ghi_clr_proxy"] - out["ghi_clr_pv_raw"]) / s_ghi)
        + charbonnier((out["dni_clr_proxy"] - out["dni_clr_pv_raw"]) / s_dni),
        clear_mask,
    )

    if stage == 1:
        # Stage-1 met-only clear-sky pretraining.
        total = (
            cfg["lambda_clr"] * l_clr
            + cfg["lambda_dt"] * l_dt
            + cfg["lambda_delta"] * l_delta
            + cfg["lambda_base"] * l_base
            + cfg["lambda_aod"] * l_aod
            + cfg.get("lambda_proxy", 0.2) * l_proxy
        )
    else:
        total = (
            l_rad
            + cfg["lambda_split"] * l_split
            + cfg["lambda_id"] * l_id
            + cfg["lambda_dt"] * l_dt
            + cfg["lambda_delta"] * l_delta
            + cfg["lambda_base"] * l_base
            + cfg["lambda_clr"] * l_clr
            + cfg["lambda_f0"] * l_f0
            + cfg["lambda_aod"] * l_aod
            + cfg["lambda_smooth"] * l_cloud_smooth
            + cfg.get("lambda_proxy", 0.05) * l_proxy
        )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_rad": float(l_rad.detach().cpu()),
        "loss_split": float(l_split.detach().cpu()),
        "loss_id": float(l_id.detach().cpu()),
        "loss_dt": float(l_dt.detach().cpu()),
        "loss_delta": float(l_delta.detach().cpu()),
        "loss_base": float(l_base.detach().cpu()),
        "loss_clr": float(l_clr.detach().cpu()),
        "loss_f0": float(l_f0.detach().cpu()),
        "loss_aod": float(l_aod.detach().cpu()),
        "loss_cloud_smooth": float(l_cloud_smooth.detach().cpu()),
        "loss_proxy": float(l_proxy.detach().cpu()),
    }
    return total, stats
