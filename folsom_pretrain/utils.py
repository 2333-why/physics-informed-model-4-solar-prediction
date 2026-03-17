from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


@dataclass
class AmpHelper:
    autocast_ctx: object
    scaler: object | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def is_npu_available() -> bool:
    return bool(torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available())


def is_npu_device(device: torch.device) -> bool:
    return device.type == "npu"


def should_pin_memory(device: torch.device, pin_memory: bool) -> bool:
    return False


def get_device(device_pref: str = "npu") -> torch.device:
    pref = device_pref.lower()

    if pref not in {"npu", "auto"}:
        raise ValueError(f"Unsupported device preference for this NPU-only project: {device_pref}")
    if not is_npu_available():
        raise RuntimeError("Requested NPU, but torch_npu is unavailable or no Ascend NPU was detected.")
    torch.npu.set_device("npu:0")
    return torch.device("npu:0")


def get_amp_helper(device: torch.device, enabled: bool = True) -> AmpHelper:
    if not enabled:
        return AmpHelper(autocast_ctx=nullcontext, scaler=None)

    if device.type == "npu":
        try:
            from torch_npu.npu.amp import GradScaler, autocast

            scaler = GradScaler()
            return AmpHelper(autocast_ctx=autocast, scaler=scaler)
        except Exception:
            return AmpHelper(autocast_ctx=nullcontext, scaler=None)

    raise RuntimeError(f"Unsupported device type for this NPU-only project: {device.type}")
