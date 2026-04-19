from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


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

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera index {self.camera_index}")

    def detect(self) -> bool:
        import cv2

        self._lazy_init()
        ok, frame = self._cap.read()
        if not ok:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
        return len(faces) > 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def build(backend: str, camera_index: int) -> FaceDetector:
    if backend == "opencv-haar":
        return OpenCVHaar(camera_index=camera_index)
    raise ValueError(f"unknown face backend: {backend!r}")
