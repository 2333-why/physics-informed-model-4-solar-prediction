from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from folsom_pretrain.data import DatasetSpec, FolsomImageSequenceDataset, collate_fn
from folsom_pretrain.losses import physics_loss
from folsom_pretrain.models import CloudPhysicsFinalModel
from folsom_pretrain.utils import (
    get_amp_helper,
    get_device,
    is_npu_device,
    seed_everything,
    should_pin_memory,
)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_clear_datasets(cfg: dict) -> tuple[FolsomImageSequenceDataset, FolsomImageSequenceDataset]:
    dcfg = cfg["dataset"]
    common = dict(
        irradiance_csv=dcfg["irradiance_csv"],
        weather_csv=dcfg["weather_csv"],
        image_root=dcfg["image_root"],
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
        clear_only=True,
    )
    train_spec = DatasetSpec(**common, split="train")
    val_spec = DatasetSpec(**common, split="val")
    return FolsomImageSequenceDataset(train_spec), FolsomImageSequenceDataset(val_spec)


def build_loader(dataset: FolsomImageSequenceDataset, cfg: dict, device: torch.device, shuffle: bool) -> DataLoader:
    train_cfg = cfg["train"]
    num_workers = train_cfg.get("num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=should_pin_memory(device, train_cfg.get("pin_memory", False)),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def run_epoch(
    model: CloudPhysicsFinalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    amp,
    device: torch.device,
    loss_cfg: dict,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    meters: dict[str, float] = {}
    n_batches = 0

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        weather = batch["weather"].to(device, non_blocking=True)
        solar = batch["solar"].to(device, non_blocking=True)
        target_rad = batch["target_rad"].to(device, non_blocking=True)
        clear_mask = batch["clear_mask"].to(device, non_blocking=True)

        if clear_mask.sum().item() <= 0:
            continue

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with amp.autocast_ctx():
            out = model(images, weather, solar)
            loss, loss_items = physics_loss(
                out=out,
                target_rad=target_rad,
                clear_mask=clear_mask,
                stage=1,
                cfg=loss_cfg,
            )

        if is_train:
            if amp.scaler is not None:
                amp.scaler.scale(loss).backward()
                amp.scaler.step(optimizer)
                amp.scaler.update()
            else:
                loss.backward()
                optimizer.step()

        n_batches += 1
        for k, v in loss_items.items():
            meters[k] = meters.get(k, 0.0) + float(v)
        pbar.set_postfix({"loss": f"{loss_items['loss_total']:.4f}", "stage": 1})

    if n_batches == 0:
        return {"loss_total": 0.0}
    return {k: v / max(1, n_batches) for k, v in meters.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 clear-sky pretraining (AOD/residual heads only)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    seed_everything(cfg.get("seed", 42))

    device = get_device(cfg.get("device", "npu"))
    if not is_npu_device(device):
        raise RuntimeError(f"This training entrypoint is configured for Ascend NPU only, but got {device}.")
    print(f"Using device: {device}")

    train_ds, val_ds = build_clear_datasets(cfg)
    train_loader = build_loader(train_ds, cfg, device, shuffle=True)
    val_loader = build_loader(val_ds, cfg, device, shuffle=False)

    model = CloudPhysicsFinalModel(
        weather_dim=7,
        feat_dim=cfg["model"].get("feat_dim", 128),
        hidden_dim=cfg["model"].get("hidden_dim", 128),
        rs=cfg["model"].get("rs", 0.2),
    ).to(device)
    model.freeze_stage1()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-3),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    amp = get_amp_helper(device, enabled=cfg["train"].get("amp", True))
    epochs = args.epochs or cfg["train"].get("stage1_epochs", 5)
    loss_cfg = cfg["loss"]
    out_dir = Path(args.out_dir or cfg["train"].get("out_dir", "outputs/final_pvlib_farms"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, amp, device, loss_cfg)
        val_m = run_epoch(model, val_loader, None, amp, device, loss_cfg)
        print(
            f"Epoch {epoch:03d} (stage 1) | "
            f"train={train_m['loss_total']:.4f} val={val_m['loss_total']:.4f} "
            f"val_clr={val_m.get('loss_clr', 0.0):.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "stage": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_m["loss_total"],
            "config": cfg,
        }
        torch.save(ckpt, out_dir / "stage1_last.pt")
        if val_m["loss_total"] < best_val:
            best_val = val_m["loss_total"]
            torch.save(ckpt, out_dir / "stage1_best.pt")


if __name__ == "__main__":
    main()
