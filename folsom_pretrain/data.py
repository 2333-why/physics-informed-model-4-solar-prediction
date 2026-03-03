from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .solar import build_solar_features


@dataclass
class DatasetSpec:
    metadata_csv: str
    image_root: str
    timestamp_col: str = "timestamp"
    image_col: str = "image_path"
    target_col: str = "ghi"
    weather_cols: tuple[str, ...] = ("air_temperature", "relative_humidity", "wind_speed")
    seq_len: int = 8
    stride: int = 1
    latitude: float = 38.5791
    longitude: float = -121.4910
    timezone: str = "US/Pacific"
    altitude: float | None = None
    image_size: int = 224


class FolsomImageSequenceDataset(Dataset):
    """
    Dataset for image-physics pretraining.

    Each sample returns a sequence of:
    - sky images
    - weather features
    - pvlib solar features
    - measured GHI labels
    """

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.image_root = Path(spec.image_root)
        self.df = pd.read_csv(spec.metadata_csv)

        required = {
            spec.timestamp_col,
            spec.image_col,
            spec.target_col,
            *spec.weather_cols,
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.df = self.df.sort_values(spec.timestamp_col).reset_index(drop=True)
        self.df[spec.timestamp_col] = pd.to_datetime(self.df[spec.timestamp_col])

        solar_df = build_solar_features(
            timestamps=self.df[spec.timestamp_col],
            latitude=spec.latitude,
            longitude=spec.longitude,
            tz=spec.timezone,
            altitude=spec.altitude,
        )
        self.df = pd.concat([self.df, solar_df], axis=1)

        self.transform = transforms.Compose(
            [
                transforms.Resize((spec.image_size, spec.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.seq_starts = list(range(0, len(self.df) - spec.seq_len + 1, spec.stride))
        if not self.seq_starts:
            raise ValueError(
                f"No valid sequence can be formed. len(df)={len(self.df)}, seq_len={spec.seq_len}"
            )

    def __len__(self) -> int:
        return len(self.seq_starts)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = self.image_root / rel_path
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.seq_starts[idx]
        end = start + self.spec.seq_len
        clip = self.df.iloc[start:end]

        images = torch.stack([self._load_image(p) for p in clip[self.spec.image_col].tolist()], dim=0)

        weather = torch.as_tensor(
            clip[list(self.spec.weather_cols)].values,
            dtype=torch.float32,
        )

        solar = torch.as_tensor(
            clip[["solar_azimuth", "solar_zenith", "cos_zenith", "cs_ghi"]].values,
            dtype=torch.float32,
        )

        target_ghi = torch.as_tensor(clip[self.spec.target_col].values, dtype=torch.float32)

        return {
            "images": images,
            "weather": weather,
            "solar": solar,
            "target_ghi": target_ghi,
        }


def collate_fn(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out
