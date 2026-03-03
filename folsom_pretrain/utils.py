from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass
class AmpHelper:
    autocast_ctx: object
    scaler: object | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_pref: str = "auto") -> torch.device:
    pref = device_pref.lower()
    if pref in {"npu", "auto"}:
        try:
            import torch_npu  # noqa: F401

            return torch.device("npu:0")
        except Exception:
            if pref == "npu":
                raise RuntimeError("Requested NPU, but torch_npu is not available.")

    if pref in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def get_amp_helper(device: torch.device, enabled: bool = True) -> AmpHelper:
    if not enabled:
        return AmpHelper(autocast_ctx=nullcontext, scaler=None)

    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        return AmpHelper(autocast_ctx=torch.cuda.amp.autocast, scaler=scaler)

    if device.type == "npu":
        try:
            from torch_npu.npu.amp import GradScaler, autocast

            scaler = GradScaler()
            return AmpHelper(autocast_ctx=autocast, scaler=scaler)
        except Exception:
            return AmpHelper(autocast_ctx=nullcontext, scaler=None)

    return AmpHelper(autocast_ctx=nullcontext, scaler=None)
