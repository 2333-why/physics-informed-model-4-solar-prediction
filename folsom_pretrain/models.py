from __future__ import annotations

import torch
import torch.nn as nn


class ImageCloudEncoder(nn.Module):
    """CNN + GRU that maps image sequences to cloud-parameter sequences."""

    def __init__(self, cloud_dim: int = 4, feat_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.cloud_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cloud_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, T, 3, H, W]
        bsz, steps, ch, h, w = images.shape
        x = images.view(bsz * steps, ch, h, w)
        x = self.backbone(x)
        x = x.view(bsz, steps, -1)
        x, _ = self.temporal(x)
        cloud_raw = self.cloud_head(x)
        cloud_params = torch.sigmoid(cloud_raw)
        return cloud_params


class PhysicalIrradianceHead(nn.Module):
    """Maps cloud + weather + solar features to irradiance reconstruction."""

    def __init__(self, cloud_dim: int, weather_dim: int, solar_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        in_dim = cloud_dim + weather_dim + solar_dim
        self.attenuation = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.offset = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, cloud: torch.Tensor, weather: torch.Tensor, solar: torch.Tensor) -> torch.Tensor:
        # cloud: [B, T, C], weather: [B, T, W], solar: [B, T, 4]
        x = torch.cat([cloud, weather, solar], dim=-1)

        # Use clear-sky GHI as physics anchor
        cs_ghi = solar[..., 3:4].clamp(min=0.0)
        attenuation = torch.sigmoid(self.attenuation(x))
        offset = torch.nn.functional.softplus(self.offset(x))
        gain = torch.nn.functional.softplus(self.scale)

        ghi_hat = gain * (cs_ghi * attenuation + offset)
        return ghi_hat.squeeze(-1)


class CloudPhysicsPretrainer(nn.Module):
    def __init__(self, weather_dim: int, cloud_dim: int = 4, feat_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.encoder = ImageCloudEncoder(cloud_dim=cloud_dim, feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.physics_head = PhysicalIrradianceHead(
            cloud_dim=cloud_dim,
            weather_dim=weather_dim,
            solar_dim=4,
            hidden_dim=hidden_dim,
        )

    def forward(self, images: torch.Tensor, weather: torch.Tensor, solar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cloud_params = self.encoder(images)
        ghi_hat = self.physics_head(cloud_params, weather, solar)
        return cloud_params, ghi_hat
