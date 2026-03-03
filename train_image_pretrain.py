from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from folsom_pretrain.data import DatasetSpec, FolsomImageSequenceDataset, collate_fn
from folsom_pretrain.losses import pretrain_loss
from folsom_pretrain.models import CloudPhysicsPretrainer
from folsom_pretrain.utils import get_amp_helper, get_device, seed_everything


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict) -> tuple[FolsomImageSequenceDataset, FolsomImageSequenceDataset]:
    common = cfg["dataset"]

    train_spec = DatasetSpec(
        metadata_csv=common["train_csv"],
        image_root=common["image_root"],
        timestamp_col=common.get("timestamp_col", "timestamp"),
        image_col=common.get("image_col", "image_path"),
        target_col=common.get("target_col", "ghi"),
        weather_cols=tuple(common["weather_cols"]),
        seq_len=common.get("seq_len", 8),
        stride=common.get("stride", 1),
        latitude=common["latitude"],
        longitude=common["longitude"],
        timezone=common.get("timezone", "US/Pacific"),
        altitude=common.get("altitude", None),
        image_size=common.get("image_size", 224),
    )

    val_spec = DatasetSpec(
        metadata_csv=common.get("val_csv", common["train_csv"]),
        image_root=common["image_root"],
        timestamp_col=train_spec.timestamp_col,
        image_col=train_spec.image_col,
        target_col=train_spec.target_col,
        weather_cols=train_spec.weather_cols,
        seq_len=train_spec.seq_len,
        stride=train_spec.stride,
        latitude=train_spec.latitude,
        longitude=train_spec.longitude,
        timezone=train_spec.timezone,
        altitude=train_spec.altitude,
        image_size=train_spec.image_size,
    )

    return FolsomImageSequenceDataset(train_spec), FolsomImageSequenceDataset(val_spec)


def run_epoch(
    model: CloudPhysicsPretrainer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    amp,
    device: torch.device,
    lambda_smooth: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    meters = {"loss_total": 0.0, "loss_recon": 0.0, "loss_smooth": 0.0}
    n_batches = 0

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        weather = batch["weather"].to(device, non_blocking=True)
        solar = batch["solar"].to(device, non_blocking=True)
        target = batch["target_ghi"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with amp.autocast_ctx():
            cloud_params, ghi_pred = model(images, weather, solar)
            loss, loss_items = pretrain_loss(
                ghi_pred=ghi_pred,
                ghi_true=target,
                cloud_params=cloud_params,
                lambda_smooth=lambda_smooth,
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
        for k in meters:
            meters[k] += loss_items[k]
        pbar.set_postfix({"loss": f"{loss_items['loss_total']:.4f}"})

    return {k: v / max(1, n_batches) for k, v in meters.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Image modality physics pretraining")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    seed_everything(cfg.get("seed", 42))

    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    train_ds, val_ds = build_dataset(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"].get("batch_size", 8),
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=cfg["train"].get("pin_memory", True),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("batch_size", 8),
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=cfg["train"].get("pin_memory", True),
        collate_fn=collate_fn,
    )

    model = CloudPhysicsPretrainer(
        weather_dim=len(cfg["dataset"]["weather_cols"]),
        cloud_dim=cfg["model"].get("cloud_dim", 4),
        feat_dim=cfg["model"].get("feat_dim", 128),
        hidden_dim=cfg["model"].get("hidden_dim", 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-3),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    amp = get_amp_helper(device, enabled=cfg["train"].get("amp", True))
    epochs = cfg["train"].get("epochs", 30)
    lambda_smooth = cfg["train"].get("lambda_smooth", 0.1)

    out_dir = Path(cfg["train"].get("out_dir", "outputs/pretrain"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, amp, device, lambda_smooth)
        val_m = run_epoch(model, val_loader, None, amp, device, lambda_smooth)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_m['loss_total']:.4f} val_loss={val_m['loss_total']:.4f} "
            f"val_recon={val_m['loss_recon']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_m["loss_total"],
            "config": cfg,
        }
        torch.save(ckpt, out_dir / "last.pt")

        if val_m["loss_total"] < best_val:
            best_val = val_m["loss_total"]
            torch.save(ckpt, out_dir / "best.pt")


if __name__ == "__main__":
    main()
