"""Measure YuNet detect() + cap.read() latency on real webcam.

200 iterations, print p50/p95 for each stage. Separates detector cost
from camera I/O cost.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2


def _percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[k]


def main() -> int:
    model = os.environ.get("PRESENCED_YUNET_MODEL")
    if not model or not Path(model).is_file():
        print(f"set PRESENCED_YUNET_MODEL to a real file; got: {model!r}")
        return 2
    cam_index = int(os.environ.get("PRESENCED_CAMERA_INDEX", "0"))
    n = int(os.environ.get("MEASURE_N", "200"))

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"cannot open camera {cam_index}")
        return 2
    cap.read()  # warm-up discard

    det = cv2.FaceDetectorYN.create(
        model=model,
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )
    size: tuple[int, int] | None = None

    read_ms: list[float] = []
    detect_ms: list[float] = []
    hits = 0
    misses = 0

    for _ in range(n):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if size != (w, h):
            det.setInputSize((w, h))
            size = (w, h)
        t2 = time.perf_counter()
        _, faces = det.detect(frame)
        t3 = time.perf_counter()
        read_ms.append((t1 - t0) * 1000.0)
        detect_ms.append((t3 - t2) * 1000.0)
        if faces is not None and len(faces) > 0:
            hits += 1
        else:
            misses += 1

    cap.release()
    total = len(read_ms)
    hit_rate = 100.0 * hits / total if total else 0.0

    def fmt(xs: list[float], label: str) -> str:
        return (
            f"{label}: n={len(xs)} "
            f"p50={_percentile(xs, 50):.2f}ms "
            f"p95={_percentile(xs, 95):.2f}ms "
            f"max={max(xs):.2f}ms"
        )

    print(fmt(read_ms, "cap.read()"))
    print(fmt(detect_ms, "yunet.detect()"))
    print(f"hit_rate: {hit_rate:.1f}% ({hits}/{total})")
    print(f"misses: {misses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
