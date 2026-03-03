from __future__ import annotations

import pandas as pd
import pvlib
import numpy as np


def build_solar_features(
    timestamps: pd.Series,
    latitude: float,
    longitude: float,
    tz: str,
    altitude: float | None = None,
) -> pd.DataFrame:
    """Compute solar geometry and clear-sky GHI with pvlib."""
    times = pd.to_datetime(timestamps)
    if times.dt.tz is None:
        times = times.dt.tz_localize(tz)
    else:
        times = times.dt.tz_convert(tz)

    solpos = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )

    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=tz,
        altitude=altitude,
    )
    clearsky = location.get_clearsky(times, model="ineichen")

    out = pd.DataFrame(index=timestamps.index)
    out["solar_azimuth"] = solpos["azimuth"].astype(float).values
    out["solar_zenith"] = solpos["zenith"].astype(float).values
    elevation = (90.0 - out["solar_zenith"]).clip(lower=0.0).astype(float)
    out["cos_zenith"] = np.cos(np.deg2rad(90.0 - elevation))
    out["cs_ghi"] = clearsky["ghi"].astype(float).values
    return out
