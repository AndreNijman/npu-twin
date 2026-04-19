# ADR-0004: YuNet ONNX over OpenCV Haar for face presence

Date: 2026-04-19
Status: Accepted

## Context

The Project B vault spec always described the detector stage as
`camera → YuNet face → head-pose/gaze → FSM`. The Phase 3 skeleton
shipped with OpenCV's classic Haar cascade instead, as a fast way to
stand up the FSM loop without pulling an ONNX model into the repo.

On-device smoke on the integrated UVC webcam (2026-04-19) exposed two
concrete weaknesses of the Haar path:

1. Haar needs `haarcascade_frontalface_default.xml` on disk. On Arch
   Linux the `python-opencv` package ships a single compiled `.so`
   with no `cv2.data` submodule, so the canonical
   `cv2.data.haarcascades` lookup raises `AttributeError`. A system
   path fallback (`/usr/share/opencv4/haarcascades/`) papers over this,
   but it is distro-specific and brittle.
2. At 640×480 under normal indoor lighting Haar (`scaleFactor=1.1`,
   `minNeighbors=3`, `minSize=(80,80)`) still missed roughly 25% of
   frames while a face was clearly in view. The 3 s grace period
   absorbed every flap, but the FSM transitioned
   `PRESENT ↔ AWAY_GRACE` every ~2 s, and `minSize` had to be tuned by
   hand to reject small background false positives.

## Decision

Replace the Haar skeleton with **YuNet ONNX** (`face_detection_yunet_2023mar`,
INT8 variant as shipped by OpenCV Zoo) as the default detector. Keep the
Haar backend behind a `PRESENCED_DETECTOR=haar` config flag for one cycle
so an on-device revert is a single env-var toggle.

## Reasoning

- YuNet is a small (~230 KB INT8) CNN face detector that OpenCV ships
  first-class via `cv2.FaceDetectorYN`. No external inference runtime,
  no Python-side preprocessing — `detector.detect(frame)` returns
  bounding boxes + landmarks directly.
- Published accuracy numbers (WIDER FACE, easy/medium/hard) are
  materially higher than cascade classifiers. Empirically, we expect
  the ~25% miss rate observed with Haar to drop under 5%, which
  removes the need for an anti-flap consensus filter on top of the
  FSM.
- The landmark output (eye / nose / mouth-corner points) is a
  prerequisite for the Phase 3b gaze layer. Starting with a detector
  that already produces landmarks avoids a second detector swap later.
- Resolves the `cv2.data` fragility: YuNet loads the ONNX path
  directly, no distro-specific cascade directory lookup needed.
- CPU cost is still well under the budget — OpenCV's DNN backend runs
  the INT8 variant at >100 fps on a single Zen 4 core, which leaves
  plenty of headroom for the 2 Hz sampler this daemon uses.

## Alternatives considered

- **MediaPipe Face Detection:** Comparable accuracy, but adds a much
  heavier runtime dependency (TFLite + protobuf + abseil) and its
  landmark output is geared toward full face mesh rather than the
  handful of points we need for gaze. Kept as optional extra for
  future experimentation but not the default.
- **Anti-flap consensus filter on top of Haar:** ~20 lines of code to
  require N-of-M samples before a state transition. Cheap, but it
  layers a workaround on top of a detector that was always a
  placeholder, and it does nothing for the landmark requirement Phase
  3b needs.
- **Custom XDNA 1 offload:** Ruled out by ADR-0002 — no supported
  inference runtime path on this hardware today.

## Consequences

- `scripts/fetch-yunet.sh` pulls the FP32 reference and the INT8
  quantized variant from OpenCV Zoo. Models are gitignored under
  `project-b/models/yunet/`; users run the script once before their
  first on-device smoke.
- Runtime dependency surface is unchanged (`opencv-python` already
  carries the DNN backend).
- Haar detector remains present for one cycle after the default flip,
  selectable via `PRESENCED_DETECTOR=haar`. A follow-up ADR will log
  its removal once YuNet has been smoke-tested on the real camera.
- Requires the on-device validation gate before Phase 3 can be marked
  complete: FSM flap rate < 5%, detector CPU < 3% of one core,
  per-frame latency < 15 ms, no `cv2.data` fragility.
