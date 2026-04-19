from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    frame_interval_s: float = 1.0
    grace_period_s: float = 30.0
    away_action: str = "dpms off"
    present_action: str = "dpms on"
    detector: str = "haar"
    gaze_enabled: bool = False
    log_level: str = "INFO"
    xdna_device: Path = field(default_factory=lambda: Path("/dev/accel/accel0"))

    @classmethod
    def from_env(cls) -> "Config":
        def f(name: str, default: float) -> float:
            v = os.environ.get(name)
            return float(v) if v else default

        def s(name: str, default: str) -> str:
            return os.environ.get(name, default)

        def i(name: str, default: int) -> int:
            v = os.environ.get(name)
            return int(v) if v else default

        def b(name: str, default: bool) -> bool:
            v = os.environ.get(name)
            if v is None:
                return default
            return v.lower() in ("1", "true", "yes", "on")

        return cls(
            camera_index=i("PRESENCED_CAMERA_INDEX", 0),
            frame_interval_s=f("PRESENCED_FRAME_INTERVAL_S", 1.0),
            grace_period_s=f("PRESENCED_GRACE_PERIOD_S", 30.0),
            away_action=s("PRESENCED_AWAY_ACTION", "dpms off"),
            present_action=s("PRESENCED_PRESENT_ACTION", "dpms on"),
            detector=s("PRESENCED_DETECTOR", "haar"),
            gaze_enabled=b("PRESENCED_GAZE_ENABLED", False),
            log_level=s("PRESENCED_LOG_LEVEL", "INFO"),
            xdna_device=Path(s("PRESENCED_XDNA_DEVICE", "/dev/accel/accel0")),
        )
