from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from folsom_pretrain.data import DatasetSpec, FolsomImageSequenceDataset, collate_fn
from folsom_pretrain.models import CloudPhysicsFinalModel
from folsom_pretrain.utils import get_device, is_npu_device, should_pin_memory


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_test_dataset(cfg: dict) -> FolsomImageSequenceDataset:
    dcfg = cfg["dataset"]
    spec = DatasetSpec(
        irradiance_csv=dcfg["irradiance_csv"],
        weather_csv=dcfg["weather_csv"],
        image_root=dcfg["image_root"],
        split="test",
        train_ratio=dcfg.get("train_ratio", 0.8),
        val_ratio=dcfg.get("val_ratio", 0.1),
        timestamp_col=dcfg.get("timestamp_col", "timeStamp"),
        latitude=dcfg["latitude"],
        longitude=dcfg["longitude"],
        timezone=dcfg.get("timezone", "US/Pacific"),
        altitude=dcfg.get("altitude", 56.0),
        seq_len=dcfg.get("seq_len", 8),
        stride=dcfg.get("stride", 1),
        image_size=dcfg.get("image_size", 224),
        min_mu0_day=dcfg.get("min_mu0_day", 0.0),
        closure_thresh_global=dcfg.get("closure_thresh_global", 0.05),
        min_mu0_clear=dcfg.get("min_mu0_clear", 0.25),
        clear_beam_frac_thresh=dcfg.get("clear_beam_frac_thresh", 0.6),
        clear_diff_frac_thresh=dcfg.get("clear_diff_frac_thresh", 0.25),
        clear_closure_thresh=dcfg.get("clear_closure_thresh", 0.05),
        clear_beam_frac_thresh_relaxed=dcfg.get("clear_beam_frac_thresh_relaxed", 0.5),
        clear_diff_frac_thresh_relaxed=dcfg.get("clear_diff_frac_thresh_relaxed", 0.35),
        clear_min_count=dcfg.get("clear_min_count", 100),
        clear_min_fraction=dcfg.get("clear_min_fraction", 0.02),
        stability_mode=dcfg.get("stability_mode", "auto"),
        stability_k_1min=dcfg.get("stability_k_1min", 10),
        stability_k_5min=dcfg.get("stability_k_5min", 6),
        stability_dni_1min=dcfg.get("stability_dni_1min", 120.0),
        stability_dhi_1min=dcfg.get("stability_dhi_1min", 60.0),
        stability_dni_5min=dcfg.get("stability_dni_5min", 200.0),
        stability_dhi_5min=dcfg.get("stability_dhi_5min", 100.0),
    )
    return FolsomImageSequenceDataset(spec)


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.mean(np.abs(y)) + 1e-6)
    nrmse = rmse / denom
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-6)
    r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "nrmse": nrmse, "r2": float(r2)}


def evaluate(model: CloudPhysicsFinalModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()

    y_ghi, y_dni, y_dhi = [], [], []
    p_ghi, p_dni, p_dhi = [], [], []
    clear_masks = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="test", leave=False):
            images = batch["images"].to(device, non_blocking=True)
            weather = batch["weather"].to(device, non_blocking=True)
            solar = batch["solar"].to(device, non_blocking=True)

            out = model(images, weather, solar)
            target = batch["target_rad"].to(device)
            cmask = batch["clear_mask"].to(device)

            y_ghi.append(target[..., 0].reshape(-1).cpu().numpy())
            y_dni.append(target[..., 1].reshape(-1).cpu().numpy())
            y_dhi.append(target[..., 2].reshape(-1).cpu().numpy())

            p_ghi.append(out["ghi_hat"].reshape(-1).cpu().numpy())
            p_dni.append(out["dni_hat"].reshape(-1).cpu().numpy())
            p_dhi.append(out["dhi_hat"].reshape(-1).cpu().numpy())

            clear_masks.append(cmask.reshape(-1).cpu().numpy())

    y_ghi = np.concatenate(y_ghi)
    y_dni = np.concatenate(y_dni)
    y_dhi = np.concatenate(y_dhi)
    p_ghi = np.concatenate(p_ghi)
    p_dni = np.concatenate(p_dni)
    p_dhi = np.concatenate(p_dhi)
    cm = np.concatenate(clear_masks) > 0.5

    out = {
        "all": {
            "GHI": _metrics(y_ghi, p_ghi),
            "DNI": _metrics(y_dni, p_dni),
            "DHI": _metrics(y_dhi, p_dhi),
        },
        "n_samples": int(y_ghi.size),
        "n_clear": int(cm.sum()),
    }

    if cm.sum() > 10:
        out["clear_only"] = {
            "GHI": _metrics(y_ghi[cm], p_ghi[cm]),
            "DNI": _metrics(y_dni[cm], p_dni[cm]),
            "DHI": _metrics(y_dhi[cm], p_dhi[cm]),
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final physics-informed model on test split")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="outputs/final_pvlib_farms/best.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = get_device(cfg.get("device", "npu"))
    if not is_npu_device(device):
        raise RuntimeError(f"This evaluation entrypoint is configured for Ascend NPU only, but got {device}.")

    test_ds = build_test_dataset(cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=should_pin_memory(device, cfg["train"].get("pin_memory", False)),
        persistent_workers=cfg["train"].get("num_workers", 4) > 0,
        collate_fn=collate_fn,
    )

    model = CloudPhysicsFinalModel(
        weather_dim=7,
        feat_dim=cfg["model"].get("feat_dim", 128),
        hidden_dim=cfg["model"].get("hidden_dim", 128),
        rs=cfg["model"].get("rs", 0.2),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    metrics = evaluate(model, test_loader, device)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out_dir = Path(cfg["train"].get("out_dir", "outputs/final_pvlib_farms"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
