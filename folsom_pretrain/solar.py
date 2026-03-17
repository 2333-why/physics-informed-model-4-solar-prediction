from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
import torch


def _to_localized_time(timestamps: pd.Series, tz: str) -> pd.DatetimeIndex:
    ts = pd.to_datetime(timestamps)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(tz)
    else:
        ts = ts.dt.tz_convert(tz)
    return pd.DatetimeIndex(ts)


def build_solar_features(
    timestamps: pd.Series,
    latitude: float,
    longitude: float,
    tz: str,
    altitude: float | None = None,
) -> pd.DataFrame:
    """Build solar geometry and stable clear-sky baseline for screening/training."""
    times = _to_localized_time(timestamps, tz=tz)

    solpos = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    # Stable baseline for clear-sky detection. Model forward uses Solis with predicted AOD/PWV.
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=tz,
        altitude=altitude,
    )
    cs = location.get_clearsky(times, model="ineichen")

    zenith = solpos["apparent_zenith"].astype(float).to_numpy()
    mu0 = np.clip(np.cos(np.deg2rad(zenith)), 0.0, 1.0)

    out = pd.DataFrame(index=timestamps.index)
    out["solar_azimuth"] = solpos["azimuth"].astype(float).to_numpy()
    out["solar_zenith"] = solpos["apparent_zenith"].astype(float).to_numpy()
    out["apparent_elevation"] = solpos["apparent_elevation"].astype(float).to_numpy()
    out["mu0"] = mu0
    out["dni_extra"] = dni_extra.astype(float).to_numpy()
    out["cs_ghi_ineichen"] = cs["ghi"].astype(float).to_numpy()
    out["cs_dni_ineichen"] = cs["dni"].astype(float).to_numpy()
    out["cs_dhi_ineichen"] = cs["dhi"].astype(float).to_numpy()
    return out


def solis_clearsky_torch(
    apparent_elevation: torch.Tensor,
    dni_extra: torch.Tensor,
    pressure_pa: torch.Tensor,
    aod700: torch.Tensor,
    pwv_cm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Non-differentiable pvlib Simplified Solis wrapper for batched tensors [B, T].
    Returns (ghi, dni, dhi) tensors on same device/dtype.
    """
    device = apparent_elevation.device
    dtype = apparent_elevation.dtype

    ae = apparent_elevation.detach().cpu().numpy().reshape(-1)
    de = dni_extra.detach().cpu().numpy().reshape(-1)
    pp = pressure_pa.detach().cpu().numpy().reshape(-1)
    aod = aod700.detach().cpu().numpy().reshape(-1)
    pwv = pwv_cm.detach().cpu().numpy().reshape(-1)

    # Guard ranges for numerical stability.
    ae = np.clip(ae, -5.0, 90.0)
    de = np.clip(de, 1200.0, 1420.0)
    pp = np.clip(pp, 7.0e4, 1.08e5)
    aod = np.clip(aod, 0.0, 0.45)
    pwv = np.clip(pwv, 0.05, 8.0)

    solis = pvlib.clearsky.simplified_solis(
        apparent_elevation=ae,
        aod700=aod,
        precipitable_water=pwv,
        pressure=pp,
        dni_extra=de,
    )

    ghi = torch.as_tensor(np.asarray(solis["ghi"]).reshape(apparent_elevation.shape), device=device, dtype=dtype)
    dni = torch.as_tensor(np.asarray(solis["dni"]).reshape(apparent_elevation.shape), device=device, dtype=dtype)
    dhi = torch.as_tensor(np.asarray(solis["dhi"]).reshape(apparent_elevation.shape), device=device, dtype=dtype)
    return ghi.clamp_min(0.0), dni.clamp_min(0.0), dhi.clamp_min(0.0)
