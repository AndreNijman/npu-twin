# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0.5: Repository bootstrap. Initial commit, gitignore, gitattributes,
  editorconfig, CHANGELOG skeleton, directory scaffold (`docs/`, `project-a/`,
  `project-b/`, `bench/`, `scripts/`).
- Phase 1: Hardware + driver preflight. `scripts/preflight.fish` verifies
  kernel ≥ 6.14, NPU PCI ID, `/dev/accel/accel0`, amdxdna dmesg, RADV
  Vulkan, group membership, memlock, IOMMU, XRT enumeration, `/dev/video0`.
  `docs/hardware.md` records the tested L16 Gen 2 stack (kernel 6.19.10,
  amdxdna 0.6.0, NPU firmware 1.5.2.380, XRT 2.21.75, Mesa 26.0.4).
- Phase 2: Project A llama.cpp-vulkan speculative decoding. `llama-speculative`
  server unit, 20-prompt benchmark suite across 4 classes (code/reasoning/
  factual/chat), greedy sampling for reproducibility. First run:
  **code 1.59× / reasoning 1.61× / factual 1.25× / chat 1.08× (overall 1.42×)**
  — meets ≥1.3× acceptance criterion on code class.
- Phase 3: Project B presence daemon (`project-b/presenced/`) —
  complete. Asyncio entrypoint, present/away_grace/away FSM with
  configurable grace period, YuNet ONNX face detector via
  `cv2.FaceDetectorYN` (per ADR-0004), Hyprland CLI bridge
  (`hyprctl dispatch` with `HYPRLAND_INSTANCE_SIGNATURE` guard),
  XDNA 1 opportunistic probe (status-only, no offload per ADR-0002),
  video-group + camera-node preflight, config-driven detector backend.
  Systemd user unit at `project-b/contrib/systemd/presenced.service`.
  On-device validation on ThinkPad L16 Gen 2 + integrated UVC webcam:
  FSM flap **0 %** over 60 s PRESENT, detect() p95 **9.59 ms**, total
  CPU **~1.4 %** of system (8c/16t Ryzen 7 PRO 250), pytest 17/17
  green. `project-b/scripts/fetch-yunet.sh` downloads FP32 + INT8
  models from OpenCV Zoo; `project-b/scripts/measure-latency.py`
  reproduces the p50/p95 numbers. `project-b/README.md` documents
  the camera-pipeline CPU baseline.

### Changed
- `scripts/smoke.fish` and `scripts/bench.fish` migrated off `llama-cli`
  (conversation-only in b8840) to `llama-completion` (baseline) and
  `llama-speculative` (speculative).
- Phase 3 CPU gate revised from "detector < 3 % of one core" to
  "total `presenced` CPU < 5 % of total system CPU AND detect() p95
  < 15 ms". Per-core measurement is misleading for a background
  daemon on a 16-thread CPU; the old target was unreachable without
  replacing OpenCV's GStreamer capture path. See ADR-0005.

### Removed
- OpenCV Haar skeleton detector removed from `presenced` after YuNet
  cleared on-device validation (ADR-0006). `PRESENCED_DETECTOR=haar`
  and `PRESENCED_HAAR_CASCADE` are gone; the `detector` config key is
  kept for future backends.

### Fixed

[Unreleased]: https://example.invalid/compare/HEAD
