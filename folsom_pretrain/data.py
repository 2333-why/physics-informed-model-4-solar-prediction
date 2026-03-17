from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pvlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .solar import build_solar_features


@dataclass
class DatasetSpec:
    irradiance_csv: str
    weather_csv: str
    image_root: str
    split: str = "train"  # train | val | test
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    timestamp_col: str = "timeStamp"
    latitude: float = 38.679
    longitude: float = -121.176
    timezone: str = "US/Pacific"
    altitude: float | None = 56.0
    seq_len: int = 8
    stride: int = 1
    image_size: int = 224

    # Global filtering
    min_mu0_day: float = 0.0
    closure_thresh_global: float = 0.05

    # Clear-sky screening (Method B)
    min_mu0_clear: float = 0.25
    clear_beam_frac_thresh: float = 0.6
    clear_diff_frac_thresh: float = 0.25
    clear_closure_thresh: float = 0.05
    clear_beam_frac_thresh_relaxed: float = 0.5
    clear_diff_frac_thresh_relaxed: float = 0.35
    clear_min_count: int = 100
    clear_min_fraction: float = 0.02

    # Stability windows
    stability_mode: str = "auto"  # auto | 1min | 5min | off
    stability_k_1min: int = 10
    stability_k_5min: int = 6
    stability_dni_1min: float = 120.0
    stability_dhi_1min: float = 60.0
    stability_dni_5min: float = 200.0
    stability_dhi_5min: float = 100.0

    # Stage-1 strict clear-only subset
    clear_only: bool = False


class FolsomImageSequenceDataset(Dataset):
    """Sequence dataset with image + weather + pvlib features + GHI/DNI/DHI targets."""

    weather_cols = (
        "air_temp",
        "relhum",
        "press",
        "windsp",
        "winddir",
        "max_windsp",
        "precipitation",
    )

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.image_root = Path(spec.image_root)

        if not (0.0 < spec.train_ratio < 1.0):
            raise ValueError("train_ratio must be in (0,1)")
        if not (0.0 <= spec.val_ratio < 1.0):
            raise ValueError("val_ratio must be in [0,1)")
        if spec.train_ratio + spec.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1")

        df = self._build_merged_frame(spec)
        df = self._add_solar_and_clear_mask(df, spec)

        # Daytime and physically valid records.
        eps = 1e-6
        closure_err = np.abs(df["ghi"] - (df["dni"] * df["mu0"] + df["dhi"])) / (np.abs(df["ghi"]) + eps)
        df = df[(df["mu0"] > spec.min_mu0_day) & (closure_err < spec.closure_thresh_global)].copy()

        if spec.clear_only:
            df = df[df["clear_mask"] > 0.5].copy()

        train_end = int(len(df) * spec.train_ratio)
        val_end = int(len(df) * (spec.train_ratio + spec.val_ratio))
        if spec.split == "train":
            df = df.iloc[:train_end]
        elif spec.split == "val":
            df = df.iloc[train_end:val_end]
        elif spec.split == "test":
            df = df.iloc[val_end:]
        else:
            raise ValueError(f"Unknown split: {spec.split}")

        df = df.reset_index(drop=True)

        self.df = df
        self.transform = transforms.Compose(
            [
                transforms.Resize((spec.image_size, spec.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.seq_starts = list(range(0, len(self.df) - spec.seq_len + 1, spec.stride))
        if not self.seq_starts:
            raise ValueError(f"No sequence formed after filtering. len={len(self.df)}, seq_len={spec.seq_len}, split={spec.split}")

    def __len__(self) -> int:
        return len(self.seq_starts)

    @staticmethod
    def _resolve_weather_csv(path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        head = p.read_text(encoding="utf-8", errors="ignore")[:128].lower()
        if "<!doctype html" in head:
            alt = Path("/mnt/nvme0/chronos-forecasting-main/dataset/Folsom_weather.csv")
            if alt.exists():
                return str(alt)
            raise ValueError(f"Weather CSV appears invalid HTML redirect: {path}")
        return path

    def _build_merged_frame(self, spec: DatasetSpec) -> pd.DataFrame:
        irr = pd.read_csv(spec.irradiance_csv)
        weather_path = self._resolve_weather_csv(spec.weather_csv)
        met = pd.read_csv(weather_path)

        irr[spec.timestamp_col] = pd.to_datetime(irr[spec.timestamp_col])
        met[spec.timestamp_col] = pd.to_datetime(met[spec.timestamp_col])

        need = {"ghi", "dni", "dhi", spec.timestamp_col}
        if not need.issubset(set(irr.columns)):
            raise ValueError(f"Irradiance CSV missing {sorted(need - set(irr.columns))}")

        for c in self.weather_cols:
            if c not in met.columns:
                raise ValueError(f"Weather CSV missing column: {c}")

        df = irr[[spec.timestamp_col, "ghi", "dni", "dhi"]].merge(
            met[[spec.timestamp_col, *self.weather_cols]], on=spec.timestamp_col, how="inner"
        )
        df = df.sort_values(spec.timestamp_col).reset_index(drop=True)

        # Build image relative path from timestamp: YYYY/MM/DD/YYYYMMDD_HHMMSS.jpg
        ts = pd.to_datetime(df[spec.timestamp_col])
        rel = ts.dt.strftime("%Y/%m/%d/%Y%m%d_%H%M%S.jpg")
        df["image_path"] = rel

        full_paths = [self.image_root / p for p in df["image_path"].tolist()]
        exists = np.array([p.exists() for p in full_paths], dtype=bool)
        df = df.loc[exists].copy().reset_index(drop=True)
        return df

    @staticmethod
    def _run_mask(step_ok: np.ndarray, min_points: int) -> np.ndarray:
        n = len(step_ok) + 1
        out = np.zeros(n, dtype=bool)
        i = 0
        while i < len(step_ok):
            if not step_ok[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(step_ok) and step_ok[j + 1]:
                j += 1
            points = (j - i + 1) + 1
            if points >= min_points:
                out[i : j + 2] = True
            i = j + 1
        return out

    def _stability_mask(self, out: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
        if spec.stability_mode.lower() == "off":
            return np.ones(len(out), dtype=bool)

        ts = pd.to_datetime(out[spec.timestamp_col])
        dt = ts.diff().dt.total_seconds().to_numpy()
        if len(dt) <= 1:
            return np.ones(len(out), dtype=bool)

        med_dt = float(np.nanmedian(dt[1:]))
        mode = spec.stability_mode.lower()
        if mode == "auto":
            mode = "1min" if med_dt <= 90.0 else "5min"

        if mode == "1min":
            k = spec.stability_k_1min
            thr_dni = spec.stability_dni_1min
            thr_dhi = spec.stability_dhi_1min
        elif mode == "5min":
            k = spec.stability_k_5min
            thr_dni = spec.stability_dni_5min
            thr_dhi = spec.stability_dhi_5min
        else:
            return np.ones(len(out), dtype=bool)

        dni = out["dni"].to_numpy(dtype=float)
        dhi = out["dhi"].to_numpy(dtype=float)
        ddni = np.abs(np.diff(dni))
        ddhi = np.abs(np.diff(dhi))
        # Block large timestamp gaps from being connected in the same stable run.
        gap_ok = dt[1:] <= max(1.5 * med_dt, 1.0)
        step_ok = (ddni < thr_dni) & (ddhi < thr_dhi) & gap_ok
        return self._run_mask(step_ok, min_points=k)

    def _add_solar_and_clear_mask(self, df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
        solar = build_solar_features(
            timestamps=df[spec.timestamp_col],
            latitude=spec.latitude,
            longitude=spec.longitude,
            tz=spec.timezone,
            altitude=spec.altitude,
        )
        out = pd.concat([df, solar], axis=1)

        # Method A: pvlib detect_clearsky with Ineichen expected GHI.
        idx = pd.DatetimeIndex(pd.to_datetime(out[spec.timestamp_col]))
        if idx.tz is None:
            idx = idx.tz_localize(spec.timezone)
        else:
            idx = idx.tz_convert(spec.timezone)

        measured = pd.Series(out["ghi"].to_numpy(), index=idx)
        expected = pd.Series(out["cs_ghi_ineichen"].to_numpy(), index=idx)
        try:
            clear_a = pvlib.clearsky.detect_clearsky(measured, expected).to_numpy(dtype=bool)
        except Exception:
            clear_a = np.zeros(len(out), dtype=bool)

        # Method B: ratio + closure rules.
        eps = 1e-6
        beam_frac = (out["dni"].to_numpy() * out["mu0"].to_numpy()) / (out["ghi"].to_numpy() + eps)
        diff_frac = out["dhi"].to_numpy() / (out["ghi"].to_numpy() + eps)
        closure = np.abs(out["ghi"].to_numpy() - (out["dni"].to_numpy() * out["mu0"].to_numpy() + out["dhi"].to_numpy())) / (
            np.abs(out["ghi"].to_numpy()) + eps
        )
        clear_b = (
            (out["mu0"].to_numpy() > spec.min_mu0_clear)
            & (beam_frac > spec.clear_beam_frac_thresh)
            & (diff_frac < spec.clear_diff_frac_thresh)
            & (closure < spec.clear_closure_thresh)
        )

        clear_b_relaxed = (
            (out["mu0"].to_numpy() > spec.min_mu0_clear)
            & (beam_frac > spec.clear_beam_frac_thresh_relaxed)
            & (diff_frac < spec.clear_diff_frac_thresh_relaxed)
            & (closure < spec.clear_closure_thresh)
        )

        stable = self._stability_mask(out, spec)

        clear_strict = clear_a & clear_b & stable
        min_keep = max(spec.clear_min_count, int(spec.clear_min_fraction * len(out)))

        candidates = [
            clear_strict,
            clear_a & clear_b,
            clear_b & stable,
            clear_a & stable,
            clear_b_relaxed & stable,
            clear_b_relaxed,
            (clear_a | clear_b) & stable,
            (clear_a | clear_b_relaxed),
        ]

        clear_final = candidates[0]
        for m in candidates:
            if int(m.sum()) >= min_keep:
                clear_final = m
                break
        else:
            clear_final = max(candidates, key=lambda m: int(m.sum()))

        out["beam_frac"] = beam_frac.astype(np.float32)
        out["diffuse_frac"] = diff_frac.astype(np.float32)
        out["clear_mask_a"] = clear_a.astype(np.float32)
        out["clear_mask_b"] = clear_b.astype(np.float32)
        out["clear_mask_b_relaxed"] = clear_b_relaxed.astype(np.float32)
        out["stable_mask"] = stable.astype(np.float32)
        out["clear_mask"] = clear_final.astype(np.float32)
        return out

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = self.image_root / rel_path
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        return self.transform(rgb)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.seq_starts[idx]
        e = s + self.spec.seq_len
        clip = self.df.iloc[s:e]

        images = torch.stack([self._load_image(p) for p in clip["image_path"].tolist()], dim=0)
        weather = torch.as_tensor(clip[list(self.weather_cols)].to_numpy(copy=True), dtype=torch.float32)

        solar = torch.as_tensor(
            clip[["solar_azimuth", "solar_zenith", "apparent_elevation", "mu0", "dni_extra"]].to_numpy(copy=True),
            dtype=torch.float32,
        )

        clear_ineichen = torch.as_tensor(
            clip[["cs_ghi_ineichen", "cs_dni_ineichen", "cs_dhi_ineichen"]].to_numpy(copy=True),
            dtype=torch.float32,
        )

        target_rad = torch.as_tensor(clip[["ghi", "dni", "dhi"]].to_numpy(copy=True), dtype=torch.float32)
        clear_mask = torch.as_tensor(clip["clear_mask"].to_numpy(copy=True), dtype=torch.float32)

        return {
            "images": images,
            "weather": weather,
            "solar": solar,
            "clear_ineichen": clear_ineichen,
            "target_rad": target_rad,
            "clear_mask": clear_mask,
        }


def collate_fn(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}
