from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


class FaceDetector(Protocol):
    def detect(self) -> bool: ...
    def close(self) -> None: ...


def _resolve_yunet_model() -> str:
    override = os.environ.get("PRESENCED_YUNET_MODEL")
    if override:
        if not Path(override).is_file():
            raise RuntimeError(f"PRESENCED_YUNET_MODEL does not exist: {override}")
        return override
    raise RuntimeError(
        "PRESENCED_YUNET_MODEL is not set; run project-b/scripts/fetch-yunet.sh "
        "and point the env var at the .onnx file"
    )


@dataclass
class YuNet:
    camera_index: int = 0
    model_path: str = ""
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 5000
    _cap: Any = field(default=None)
    _det: Any = field(default=None)
    _size: tuple[int, int] | None = field(default=None)

    def _lazy_init(self) -> None:
        if self._cap is not None:
            return
        import cv2

        model = self.model_path or _resolve_yunet_model()
        log.debug("yunet model resolved: %s", model)
        self._det = cv2.FaceDetectorYN.create(
            model=model,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera index {self.camera_index}")
        # Discard first frame: many UVC cameras return a dark/garbage frame on open.
        self._cap.read()

    def detect(self) -> bool:
        self._lazy_init()
        ok, frame = self._cap.read()
        if not ok:
            return False
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._det.detect(frame)
        return faces is not None and len(faces) > 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._det = None
        self._size = None


def build(backend: str, camera_index: int) -> FaceDetector:
    if backend == "yunet":
        return YuNet(camera_index=camera_index)
    raise ValueError(
        f"unknown detector backend: {backend!r} (expected 'yunet')"
    )
