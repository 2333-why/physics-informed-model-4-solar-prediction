from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .solar import solis_clearsky_torch


class ImageCloudEncoder(nn.Module):
    def __init__(self, feat_dim: int = 128, hidden_dim: int = 128):
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
        self.temporal = nn.GRU(feat_dim, hidden_dim, num_layers=1, batch_first=True).half()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = images.shape
        x = self.backbone(images.view(b * t, c, h, w)).view(b, t, -1)
        # Ascend DynamicGRU kernel on this stack requires fp16 weights/inputs.
        x, _ = self.temporal(x.to(dtype=self.temporal.weight_ih_l0.dtype))
        return x.float()


class CloudHead(nn.Module):
    """Outputs f, tau_cld, D_e, p_ice, p_occ."""

    def __init__(self, in_dim: int, dmin: float = 5.0, dmax: float = 120.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, 5),
        )
        self.dmin = dmin
        self.dmax = dmax

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(x)
        f = torch.sigmoid(raw[..., 0])
        tau_cld = F.softplus(raw[..., 1])
        de = self.dmin + (self.dmax - self.dmin) * torch.sigmoid(raw[..., 2])
        p_ice = torch.sigmoid(raw[..., 3])
        p_occ = torch.sigmoid(raw[..., 4])
        return {"f": f, "tau_cld": tau_cld, "de": de, "p_ice": p_ice, "p_occ": p_occ}


class AODHead(nn.Module):
    def __init__(self, weather_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        # Solis typical AOD range
        return 0.45 * torch.sigmoid(self.net(weather).squeeze(-1))


class ResidualClearSkyHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

        # Learnable baseline scalars.
        self.a0 = nn.Parameter(torch.tensor(0.10))
        self.a1 = nn.Parameter(torch.tensor(0.80))
        self.a2 = nn.Parameter(torch.tensor(0.60))
        self.b0 = nn.Parameter(torch.tensor(1.00))
        self.c0 = nn.Parameter(torch.tensor(1.00))
        self.c_sca = nn.Parameter(torch.tensor(0.70))
        self.c_abs = nn.Parameter(torch.tensor(0.20))
        self.k_ray = nn.Parameter(torch.tensor(1.00))
        self.k_aod = nn.Parameter(torch.tensor(0.85))
        self.k_w = nn.Parameter(torch.tensor(0.35))
        self.u0 = nn.Parameter(torch.tensor(0.15))
        self.u1 = nn.Parameter(torch.tensor(1.20))
        self.u2 = nn.Parameter(torch.tensor(0.60))

    def baseline_params(self) -> list[torch.Tensor]:
        return [
            self.a0,
            self.a1,
            self.a2,
            self.b0,
            self.c0,
            self.c_sca,
            self.c_abs,
            self.k_ray,
            self.k_aod,
            self.k_w,
            self.u0,
            self.u1,
            self.u2,
        ]


class CloudPhysicsFinalModel(nn.Module):
    """Final Version-B style model with pvlib clear-sky + residual clear-sky internals + FARMS-like layer."""

    def __init__(self, weather_dim: int = 7, feat_dim: int = 128, hidden_dim: int = 128, rs: float = 0.2):
        super().__init__()
        self.encoder = ImageCloudEncoder(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.cloud_head = CloudHead(in_dim=hidden_dim)
        self.aod_head = AODHead(weather_dim=weather_dim, hidden_dim=hidden_dim // 2)
        self.res_clear = ResidualClearSkyHead(in_dim=11, hidden_dim=hidden_dim // 2)
        self.delta_pwv = nn.Parameter(torch.tensor(0.0))
        self.rs = rs
        self.eps = 1e-6

    @staticmethod
    def _straight_through(target: torch.Tensor, proxy: torch.Tensor) -> torch.Tensor:
        return target + (proxy - proxy.detach())

    def _pwv_emp_cm(self, t_c: torch.Tensor, rh: torch.Tensor, p_hpa: torch.Tensor) -> torch.Tensor:
        # Simple physically monotone approximation in cm.
        t_k = t_c + 273.15
        es = 6.112 * torch.exp(17.67 * t_c / (t_c + 243.5))
        e = (rh.clamp(1.0, 100.0) / 100.0) * es
        pwv_mm = 0.1 * (e * t_k / p_hpa.clamp_min(700.0))
        pwv_cm = 0.1 * pwv_mm
        return pwv_cm.clamp(0.05, 8.0)

    def _farms_cloud_terms(self, tau_cld: torch.Tensor, de: torch.Tensor, mu0: torch.Tensor, p_ice: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FARMS Eq.(11)-(14) with fixed coefficients from the paper.
        mu = mu0.clamp_min(0.03)
        tau = tau_cld.clamp_min(0.0)
        tdd = torch.exp(-tau / mu)

        b0, b1, b2 = 0.149, 0.199, -0.036
        c1, c2 = 0.681, 0.111
        base = (b0 + b1 * mu + b2 * mu * mu) * tau / (1.0 + c1 * tau + c2 * tau * tau)

        # Particle-size adjustment F(De) for water/ice clouds.
        fw = 1.0 + 0.008 * (de - 10.0) + (-0.00003) * (de - 10.0) ** 2
        fi = 1.0 + 0.004 * (de - 30.0) + (-0.00002) * (de - 30.0) ** 2
        fw = fw.clamp_min(0.0)
        fi = fi.clamp_min(0.0)
        tdu_w = base * fw
        tdu_i = base * fi
        tdu = (1.0 - p_ice) * tdu_w + p_ice * tdu_i

        rw = (0.355 * tau) / (1.0 + 0.452 * tau)
        ri = (0.300 * tau) / (1.0 + 0.410 * tau)
        ruu = (1.0 - p_ice) * rw + p_ice * ri
        return tdd.clamp(0.0, 1.0), tdu.clamp(0.0, 1.0), ruu.clamp(0.0, 1.0)

    def _clear_sky_proxy(
        self,
        mu0: torch.Tensor,
        f0: torch.Tensor,
        tau_r: torch.Tensor,
        tau_w: torch.Tensor,
        aod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g = 1.0 / (mu0.clamp_min(0.05) + self.eps)
        tau_direct = (
            F.softplus(self.res_clear.k_ray) * tau_r
            + F.softplus(self.res_clear.k_aod) * aod
            + F.softplus(self.res_clear.k_w) * tau_w
        )
        tdd_proxy = torch.exp(-tau_direct * g).clamp(0.0, 1.0)

        tau_sca = tau_r + F.softplus(self.res_clear.c_sca) * aod
        tau_abs = tau_w + F.softplus(self.res_clear.c_abs) * aod
        tdu_proxy = torch.sigmoid(
            self.res_clear.u0
            + self.res_clear.u1 * tau_sca * g
            - self.res_clear.u2 * tau_abs * g
        ) * (1.0 - tdd_proxy)

        dni_proxy = (f0 * tdd_proxy).clamp_min(0.0)
        ghi_proxy = (mu0 * f0 * (tdd_proxy + tdu_proxy)).clamp_min(0.0)
        dhi_proxy = (ghi_proxy - dni_proxy * mu0).clamp_min(0.0)
        return ghi_proxy, dni_proxy, dhi_proxy, tdd_proxy

    def forward(self, images: torch.Tensor, weather: torch.Tensor, solar: torch.Tensor) -> dict[str, torch.Tensor]:
        # solar: [azimuth, zenith, apparent_elevation, mu0, dni_extra]
        z = self.encoder(images)
        cloud = self.cloud_head(z)

        f = cloud["f"]
        tau_cld = cloud["tau_cld"]
        de = cloud["de"]
        p_ice = cloud["p_ice"]
        p_occ = cloud["p_occ"]

        # weather columns: [air_temp, relhum, press(hPa), windsp, winddir, max_windsp, precipitation]
        t_c = weather[..., 0]
        rh = weather[..., 1]
        p_hpa = weather[..., 2].clamp(700.0, 1100.0)
        ws = weather[..., 3]
        wd = weather[..., 4]
        prcp = weather[..., 6]

        apparent_elevation = solar[..., 2]
        mu0 = solar[..., 3].clamp(0.0, 1.0)
        dni_extra = solar[..., 4].clamp_min(1200.0)
        f0 = dni_extra

        aod = self.aod_head(weather)
        pwv_cm = self._pwv_emp_cm(t_c, rh, p_hpa) * torch.exp(self.delta_pwv)
        pwv_cm = pwv_cm.clamp(0.05, 8.0)

        tau_r = 0.1 * (p_hpa / 1013.25)
        tau_w = 0.08 * pwv_cm
        ghi_clr_proxy, dni_clr_proxy, dhi_clr_proxy, tdd_proxy = self._clear_sky_proxy(
            mu0=mu0,
            f0=f0,
            tau_r=tau_r,
            tau_w=tau_w,
            aod=aod,
        )

        ghi_clr_pv_raw, dni_clr_pv_raw, dhi_clr_pv_raw = solis_clearsky_torch(
            apparent_elevation=apparent_elevation,
            dni_extra=dni_extra,
            pressure_pa=p_hpa * 100.0,
            aod700=aod,
            pwv_cm=pwv_cm,
        )

        ghi_clr_pv = self._straight_through(ghi_clr_pv_raw, ghi_clr_proxy)
        dni_clr_pv = self._straight_through(dni_clr_pv_raw, dni_clr_proxy)
        dhi_clr_pv = self._straight_through(dhi_clr_pv_raw, dhi_clr_proxy)

        tdd_clr = (dni_clr_pv / f0.clamp_min(self.eps)).clamp(0.0, 1.0)

        # AOD-based baseline internals.
        g = 1.0 / (mu0 + self.eps)
        c_sca = F.softplus(self.res_clear.c_sca)
        c_abs = F.softplus(self.res_clear.c_abs)
        tau_sca = tau_r + c_sca * aod
        tau_abs = tau_w + c_abs * aod
        tau_clr = tau_r + tau_w + aod

        tdu_base = torch.sigmoid(self.res_clear.a0 + self.res_clear.a1 * tau_sca * g - self.res_clear.a2 * tau_abs * g) * (
            1.0 - tdd_clr
        )
        tuu_base = torch.exp(-F.softplus(self.res_clear.b0) * tau_clr * g)
        ruu_base = 1.0 - torch.exp(-F.softplus(self.res_clear.c0) * tau_sca * g)

        x_phys = torch.stack(
            [mu0, p_hpa / 1013.25, pwv_cm, aod, tdd_clr, ghi_clr_pv / 1200.0, dni_clr_pv / 1200.0, dhi_clr_pv / 600.0, ws / 20.0, wd / 360.0, prcp / 10.0],
            dim=-1,
        )
        d = self.res_clear.net(x_phys)
        d_tdu, d_tuu, d_ruu = d[..., 0], d[..., 1], d[..., 2]

        tdu_clr = (tdu_base + d_tdu).clamp(0.0, 1.0)
        tuu_clr = (tuu_base + d_tuu).clamp(0.0, 1.0)
        ruu_clr = (ruu_base + d_ruu).clamp(0.0, 1.0)
        tdt_clr = tdd_clr + tdu_clr

        # FARMS overcast branch.
        tdd_cld, tdu_cld, ruu_cld = self._farms_cloud_terms(tau_cld=tau_cld, de=de, mu0=mu0, p_ice=p_ice)

        fd = mu0 * f0 * tdd_cld * tdd_clr
        dni_cld = f0 * tdd_cld * tdd_clr
        f1 = mu0 * f0 * tdd_cld * tdt_clr + mu0 * f0 * tdu_cld * tuu_clr

        denom = 1.0 - self.rs * (ruu_clr + ruu_cld * (tuu_clr**2))
        f_total = f1 / denom.clamp_min(0.05)

        ghi_cld = f_total.clamp_min(0.0)
        dhi_cld = (f_total - fd).clamp_min(0.0)

        # Partial-cloud mixing with pvlib clear-sky baseline.
        ghi_hat = f * ghi_cld + (1.0 - f) * ghi_clr_pv
        dni_hat = f * dni_cld + (1.0 - f) * dni_clr_pv
        # optional occlusion-aware gate (soft, not hard-coded):
        # dni_hat = dni_hat * (1.0 - 0.3 * p_occ)
        dhi_hat = (ghi_hat - dni_hat * mu0).clamp_min(0.0)

        return {
            "ghi_hat": ghi_hat,
            "dni_hat": dni_hat,
            "dhi_hat": dhi_hat,
            "ghi_cld": ghi_cld,
            "dni_cld": dni_cld,
            "dhi_cld": dhi_cld,
            "ghi_clr_pv": ghi_clr_pv,
            "dni_clr_pv": dni_clr_pv,
            "dhi_clr_pv": dhi_clr_pv,
            "ghi_clr_pv_raw": ghi_clr_pv_raw,
            "dni_clr_pv_raw": dni_clr_pv_raw,
            "dhi_clr_pv_raw": dhi_clr_pv_raw,
            "ghi_clr_proxy": ghi_clr_proxy,
            "dni_clr_proxy": dni_clr_proxy,
            "dhi_clr_proxy": dhi_clr_proxy,
            "tdd_proxy": tdd_proxy,
            "tdd_clr": tdd_clr,
            "tdu_clr": tdu_clr,
            "tuu_clr": tuu_clr,
            "ruu_clr": ruu_clr,
            "tdt_clr": tdt_clr,
            "fd": fd,
            "f": f,
            "p_occ": p_occ,
            "tau_cld": tau_cld,
            "aod": aod,
            "mu0": mu0,
            "d_tdu": d_tdu,
            "d_tuu": d_tuu,
            "d_ruu": d_ruu,
            "base_params_l2": sum((p * p).sum() for p in self.res_clear.baseline_params()),
        }

    def freeze_stage1(self) -> None:
        # Stage-1: met-only clear-sky modules trainable.
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.cloud_head.parameters():
            p.requires_grad = False
        for p in self.aod_head.parameters():
            p.requires_grad = True
        for p in self.res_clear.parameters():
            p.requires_grad = True
        self.delta_pwv.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
