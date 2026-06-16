"""XDNA 1 NPU opportunistic probe.

Per ADR-0002, XDNA 1 had no usable Linux inference runtime in 2026-04 —
but ADR-0007 (2026-06) supersedes that premise: the open IRON/mlir-aie/Peano
stack runs hand-written kernels on the NPU today, no VitisAI (see project-c/).
This module does NOT offload face/gaze ops. It reports whether the
device node exists, whether XRT is installed, and — when invoked via
`presenced --probe-npu` — whether onnxruntime can actually put any op
on the NPU via the VitisAIExecutionProvider.

Two entry points:

- `probe()` — lightweight existence check. Called on every startup.
- `deep_probe()` — heavy probe: parses `xrt-smi examine`, attempts an
  `onnxruntime.InferenceSession` against the Identity probe model
  shipped at project-b/models/probe.onnx, runs a dummy forward pass,
  and reports which providers were actually selected.

When AMD ships a working VitisAI EP for XDNA 1 on Linux, the deep-probe
verdict flips from `cpu-fallback` / `onnxruntime-unavailable` /
`vitis-ep-unavailable` → `npu-active`. That flip is the only
deliverable of `deep_probe()`.

Separately, the lightweight `probe()` reports `npu-active-open-stack` when
the open IRON/mlir-aie userspace is installed (`pyxrt` importable + `xrt-smi`
present + device node) — that path runs kernels on the NPU today without
VitisAI. The probe is agnostic to *which* userspace drives the device.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_PROBE_MODEL = Path(__file__).resolve().parent.parent / "models" / "probe.onnx"


@dataclass(frozen=True)
class XDNAStatus:
    device_present: bool
    device_path: Path
    xrt_cli_present: bool
    vitis_ep_importable: bool
    iron_runtime_importable: bool
    verdict: str


@dataclass
class DeepProbeResult:
    ts: str
    xrt_version: str | None
    fw_version: str | None
    device_present: bool
    providers_tried: list[str] = field(default_factory=list)
    providers_active: list[str] = field(default_factory=list)
    offloaded_ops: int = 0
    fallback_ops: int = 0
    verdict: str = "unknown"
    error: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, separators=(",", ":"))


def probe(device_path: Path = Path("/dev/accel/accel0")) -> XDNAStatus:
    device_present = device_path.exists()
    xrt = shutil.which("xrt-smi") is not None or shutil.which("xbutil") is not None
    vitis = importlib.util.find_spec("onnxruntime_vitisai") is not None
    # Open IRON/mlir-aie stack: pyxrt (the XRT Python binding) being importable
    # means the open userspace can drive the NPU directly (ADR-0007), the path
    # that actually runs kernels on XDNA 1 on Linux today.
    iron = importlib.util.find_spec("pyxrt") is not None

    if vitis and device_present:
        verdict = "runtime-available"
    elif iron and xrt and device_present:
        verdict = "npu-active-open-stack"
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
        iron_runtime_importable=iron,
        verdict=verdict,
    )


def _xrt_smi_examine() -> tuple[str | None, str | None]:
    """Return (xrt_version, firmware_version) parsed from `xrt-smi examine`.

    `xrt-smi examine` prints its version banner on stderr and the device
    table on stdout. On boxes where memlock is too low the device-table
    read fails with mmap EAGAIN (see
    errors-and-fixes/xrt-smi-mmap-memlock-eagain.md) but the version
    banner still comes through, so we still get a useful xrt_version.
    firmware_version is only readable on healthy XRT invocations.
    """
    if shutil.which("xrt-smi") is None:
        return (None, None)
    try:
        proc = subprocess.run(
            ["xrt-smi", "examine"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return (None, None)

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    xrt_v = None
    fw_v = None
    # Banner (always present): "XRT build version: 2.21.75"
    m = re.search(r"XRT\s+build\s+version\s*:\s*(\S+)", text, re.IGNORECASE)
    if m:
        xrt_v = m.group(1)
    # Fall back to per-device version row: "Version              : 2.21.75"
    if xrt_v is None:
        m = re.search(r"^\s*Version\s*:\s*(\S+)", text, re.MULTILINE)
        if m:
            xrt_v = m.group(1)
    # Firmware row (only when the device table renders):
    #   "Firmware Version     : 1.5.2.380"
    m = re.search(r"Firmware\s*Version\s*:\s*(\S+)", text)
    if m:
        fw_v = m.group(1)
    return (xrt_v, fw_v)


def deep_probe(model_path: Path = DEFAULT_PROBE_MODEL) -> tuple[DeepProbeResult, int]:
    """Run the deep probe. Returns (result, exit_code).

    Exit codes:
      0 — at least one op offloaded to the NPU (VitisAI EP active).
      1 — onnxruntime or VitisAI EP not importable (cannot probe).
      2 — EP available but every op fell back to CPU.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    xrt_v, fw_v = _xrt_smi_examine()
    device_present = Path("/dev/accel/accel0").exists()
    r = DeepProbeResult(
        ts=ts, xrt_version=xrt_v, fw_version=fw_v, device_present=device_present,
    )

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as e:
        r.verdict = "onnxruntime-unavailable"
        r.error = f"ImportError: {e}"
        return r, 1

    available = list(ort.get_available_providers())
    r.providers_tried = ["VitisAIExecutionProvider", "CPUExecutionProvider"]

    if "VitisAIExecutionProvider" not in available:
        r.verdict = "vitis-ep-unavailable"
        r.providers_active = available
        r.error = f"available providers: {available}"
        return r, 1

    if not model_path.exists():
        r.verdict = "probe-model-missing"
        r.error = f"{model_path} not found; run project-b/scripts/build-probe-onnx.py"
        return r, 1

    try:
        sess = ort.InferenceSession(
            str(model_path),
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as e:
        r.verdict = "session-create-failed"
        r.error = f"{type(e).__name__}: {e}"
        return r, 1

    r.providers_active = list(sess.get_providers())

    try:
        import numpy as np  # type: ignore[import-not-found]
        x = np.zeros((1, 3, 64, 64), dtype=np.float32)
        sess.run(None, {"x": x})
    except Exception as e:
        r.verdict = "inference-failed"
        r.error = f"{type(e).__name__}: {e}"
        return r, 1

    # onnxruntime does not expose a public per-op placement API. Heuristic:
    # if VitisAIExecutionProvider is first in the active list AND the
    # session was created without falling back, treat as NPU-active.
    # Otherwise CPU-fallback.
    if r.providers_active and r.providers_active[0] == "VitisAIExecutionProvider":
        r.verdict = "npu-active"
        r.offloaded_ops = 1
        r.fallback_ops = 0
        return r, 0

    r.verdict = "cpu-fallback"
    r.offloaded_ops = 0
    r.fallback_ops = 1
    return r, 2
