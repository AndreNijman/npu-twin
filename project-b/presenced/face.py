from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

_HAAR_FILENAME = "haarcascade_frontalface_default.xml"
_HAAR_SEARCH_PATHS = (
    "/usr/share/opencv4/haarcascades",
    "/usr/share/opencv/haarcascades",
    "/usr/local/share/opencv4/haarcascades",
)


def _resolve_haar_cascade() -> str:
    override = os.environ.get("PRESENCED_HAAR_CASCADE")
    if override:
        if not Path(override).is_file():
            raise RuntimeError(f"PRESENCED_HAAR_CASCADE does not exist: {override}")
        return override

    import cv2

    data_mod = getattr(cv2, "data", None)
    if data_mod is not None:
        candidate = Path(data_mod.haarcascades) / _HAAR_FILENAME
        if candidate.is_file():
            return str(candidate)

    for base in _HAAR_SEARCH_PATHS:
        candidate = Path(base) / _HAAR_FILENAME
        if candidate.is_file():
            return str(candidate)

    raise RuntimeError(
        f"cannot locate {_HAAR_FILENAME}; set PRESENCED_HAAR_CASCADE to override"
    )


class FaceDetector(Protocol):
    def detect(self) -> bool: ...
    def close(self) -> None: ...


@dataclass
class OpenCVHaar:
    camera_index: int = 0
    _cap: Any = field(default=None)
    _cascade: Any = field(default=None)

    def _lazy_init(self) -> None:
        if self._cap is not None:
            return
        import cv2

        cascade_path = _resolve_haar_cascade()
        log.debug("haar cascade resolved: %s", cascade_path)
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"haar cascade failed to load: {cascade_path}")
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera index {self.camera_index}")
        # Discard first frame: many UVC cameras return a dark/garbage frame on open.
        self._cap.read()

    def detect(self) -> bool:
        import cv2

        self._lazy_init()
        ok, frame = self._cap.read()
        if not ok:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80)
        )
        return len(faces) > 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def build(backend: str, camera_index: int) -> FaceDetector:
    if backend == "opencv-haar":
        return OpenCVHaar(camera_index=camera_index)
    raise ValueError(f"unknown face backend: {backend!r}")
