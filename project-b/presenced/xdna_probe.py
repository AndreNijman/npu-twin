"""XDNA 1 NPU opportunistic probe.

Per ADR-0002, XDNA 1 has no usable Linux inference runtime in 2026-04.
This module does NOT offload face/gaze ops. It reports whether the
device node exists and whether a runtime has materialized since last
boot. When AMD ships one, the status string flips — that is the only
deliverable of this probe.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class XDNAStatus:
    device_present: bool
    device_path: Path
    xrt_cli_present: bool
    vitis_ep_importable: bool
    verdict: str


def probe(device_path: Path = Path("/dev/accel/accel0")) -> XDNAStatus:
    device_present = device_path.exists()
    xrt = shutil.which("xrt-smi") is not None or shutil.which("xbutil") is not None
    vitis = importlib.util.find_spec("onnxruntime_vitisai") is not None

    if vitis and device_present:
        verdict = "runtime-available"
    elif xrt and device_present:
        verdict = "xrt-only-no-inference-runtime"
    elif device_present:
        verdict = "device-only-no-userspace"
    else:
        verdict = "no-device"

    return XDNAStatus(
        device_present=device_present,
        device_path=device_path,
        xrt_cli_present=xrt,
        vitis_ep_importable=vitis,
        verdict=verdict,
    )
