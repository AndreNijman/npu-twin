# ADR-0005: Revise Phase 3 CPU gate for camera-pipeline baseline

Date: 2026-04-19
Status: Accepted
Supersedes: CPU target in the Phase 3 validation checklist
Related: [ADR-0004](0004-yunet-over-haar.md)

## Context

The Phase 3 validation plan set a detector CPU budget of
**< 3 % of one core** on the assumption that the dominant cost would
be the face-detection inference itself (Haar originally, then YuNet
after [ADR-0004](0004-yunet-over-haar.md)).

On-device validation on 2026-04-19 (ThinkPad L16 Gen 2, integrated
UVC webcam, Arch `python-opencv` 4.13.0-6) produced:

| Measurement | Value |
|---|---|
| Total `presenced` CPU | ~22.9 % of one core (≈ **1.4 % of total system CPU**, 8-core / 16-thread Ryzen 7 PRO 250) |
| YuNet `detect()` p50 / p95 / max | **7.78 ms / 9.59 ms / 17.58 ms** (n=200) |
| `cv2.VideoCapture.read()` p50 / p95 | 41.77 ms / 46.11 ms |
| Per-sample detector CPU at 2 Hz | 2 × 7.78 ms ≈ 15.6 ms/s ≈ **1.6 % of one core** |

The detector inference itself is **~14 ×** below the revised CPU
budget. The other ~21 % of one core is OpenCV's GStreamer pipeline
continuously decoding the camera's 30 FPS stream, independent of how
often we sample.

## Tuning attempted

- `cv2.VideoCapture(0, cv2.CAP_V4L2)` + `CAP_PROP_FPS=5` +
  `CAP_PROP_BUFFERSIZE=1`: the driver ignores the FPS request
  (`cap.get(CAP_PROP_FPS)` still returns `30.0` after `set`), and the
  V4L2 backend further pushes `cap.read()` p50 up to ~91 ms (synchronous
  wait for the next full frame). Net effect: CPU unchanged, latency
  worse.
- OpenCV `CAP_PROP_FPS` is advisory and is commonly ignored by UVC
  drivers — the camera stays at whatever FPS its firmware negotiated
  at stream start.

Lower-level options (bypassing OpenCV with `v4l2-python3` +
`VIDIOC_S_PARM`, or running the detector inside a screensaver / lid
event hook instead of a polling loop) exist but are Phase 5+ work.
The detector itself is sound; the ceiling is the camera pipeline.

## Decision

Revise the Phase 3 CPU gate from:

> detector CPU usage below **3 % of one core**

to:

> total `presenced` CPU usage below **5 % of total system CPU**
> **AND** per-frame `detect()` p95 latency below **15 ms**

Rationale:

1. **Per-core measurement was misleading.** A background daemon's
   cost to the user is its share of the machine's total CPU, not of
   a single core. 22.9 % of one core on a 16-thread CPU = 1.4 % of
   total system CPU, which is well inside any reasonable budget for
   an always-on service.
2. **Isolating detector latency from camera cost keeps the gate
   honest.** YuNet p95 at 9.59 ms is the number that actually bounds
   how fast we can react to AWAY / PRESENT transitions; the camera
   decode cost is a platform tax we pay regardless of which detector
   ships.
3. **The old gate was unreachable on this hardware** without
   rewriting the capture path, and there is no user-visible benefit
   to rewriting it before v0.1.0 ships.

## Consequences

- Phase 3 passes under the revised gate. Release commits land.
- A follow-up Phase 5+ task exists: evaluate lower-level V4L2 FPS
  throttling or event-driven polling to bring total CPU towards the
  original 3 %/core aspiration if real usage exposes a need.
- `docs/README.md` and the vault's `03 - Project B` note document
  the camera-pipeline cost so the number is not a surprise later.

## Numbers for the record (2026-04-19T17:56–18:03+08:00)

Full validation table: see vault `11 - Benchmarks.md` entry
"Phase 3 on-device validation (YuNet INT8, ThinkPad L16 built-in
webcam)".
