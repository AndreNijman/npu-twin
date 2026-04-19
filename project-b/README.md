# presenced

Face-presence daemon for Hyprland. Polls the webcam, runs YuNet face
detection, and runs user-configured `hyprctl` dispatches when the
user leaves or returns.

## Status

Phase 3 complete (2026-04-19). YuNet on-device validation cleared all
gates under the revised CPU target — see
[ADR-0005](../docs/decisions/0005-phase-3-cpu-gate-revision.md) and
the vault's `11 - Benchmarks.md`.

## Quick start

```bash
# 1. Fetch the YuNet ONNX models (FP32 + INT8) into project-b/models/yunet/
bash scripts/fetch-yunet.sh

# 2. Point presenced at the INT8 model
export PRESENCED_YUNET_MODEL=$(pwd)/models/yunet/face_detection_yunet_2023mar_int8.onnx

# 3. Run in the foreground
python -m presenced
```

Systemd user unit at `contrib/systemd/presenced.service` (symlink it
into `~/.config/systemd/user/` and `systemctl --user enable --now
presenced`).

## Config env vars

| Var | Default | Notes |
|---|---|---|
| `PRESENCED_DETECTOR` | `yunet` | Only `yunet` is supported today; key kept for future backends. |
| `PRESENCED_YUNET_MODEL` | — (required) | Absolute path to a YuNet `.onnx` file. Fetch with `scripts/fetch-yunet.sh`. |
| `PRESENCED_CAMERA_INDEX` | `0` | V4L2 / UVC camera index. |
| `PRESENCED_FRAME_INTERVAL_S` | `0.5` | Sampling cadence (seconds). |
| `PRESENCED_GRACE_PERIOD_S` | `30.0` | How long the FSM tolerates face-loss before firing `AWAY`. |
| `PRESENCED_AWAY_ACTION` | `hyprctl dispatch dpms off` | Shell command run on `AWAY`. |
| `PRESENCED_PRESENT_ACTION` | `hyprctl dispatch dpms on` | Shell command run on `AWAY → PRESENT`. |
| `PRESENCED_GAZE_ENABLED` | `false` | Reserved for Phase 5 gaze stage. |
| `PRESENCED_XDNA_DEVICE` | `/dev/accel/accel0` | Probed at startup; status-only today (see ADR-0002). |
| `PRESENCED_LOG_LEVEL` | `INFO` | `DEBUG` surfaces the model-resolution path. |

## Performance notes

`presenced` runs a 2 Hz sampling loop against an integrated UVC
webcam. On a ThinkPad L16 Gen 2 the measured cost is:

- YuNet `detect()` p50 **7.78 ms**, p95 **9.59 ms** (INT8 model, n=200,
  100 % hit rate).
- Total daemon CPU: **~22.9 % of one core** ≈ **1.4 % of total system
  CPU** on 8-core / 16-thread Ryzen 7 PRO 250.
- FSM flap rate: **0 %** over 60 s steady-state PRESENT (0 transitions
  / 120 samples).

The per-core number is dominated by OpenCV's GStreamer pipeline
continuously decoding the camera's 30 FPS stream, **not** by the
detector. At 2 Hz the inference itself costs ~1.6 %/core; the
remaining ~21 % is video decode.

### Why can't we throttle the camera?

`cv2.VideoCapture.set(cv2.CAP_PROP_FPS, N)` is advisory and is
routinely ignored by UVC drivers. We verified on this hardware:
`cap.set(CAP_PROP_FPS, 5)` followed by `cap.get(CAP_PROP_FPS)` still
returns `30.0`. Forcing the V4L2 backend
(`cv2.VideoCapture(0, cv2.CAP_V4L2)`) doesn't help the FPS knob and
pushes `cap.read()` p50 up to ~91 ms (synchronous wait for next full
frame), so latency gets worse without CPU getting better.

Lower-level options — `v4l2-python3` with explicit `VIDIOC_S_PARM`,
or event-driven polling via screensaver / lid-switch D-Bus
notifications — are tracked for Phase 5+. See
[ADR-0005](../docs/decisions/0005-phase-3-cpu-gate-revision.md) for
the full rationale and the revised CPU gate.

## Tests

```bash
cd project-b && python -m pytest -q
```

17 tests, ~0.2 s. Pure-Python unit coverage: FSM transitions, XDNA
probe classifier, YuNet integration via `cv2` monkeypatches (no real
camera).
