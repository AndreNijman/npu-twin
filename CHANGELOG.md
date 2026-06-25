# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 8 / `project-c/`: **a hand-written kernel runs on the XDNA 1 "Phoenix"
  NPU** under Linux via the fully open IRON/mlir-aie/Peano stack — no Vitis, no
  VitisAI EP. Milestone **M1** (`passthrough_kernel` → `PASS!`, ~125 µs on-NPU)
  reached on Void/kernel-7.0.11 with XRT 2.25.0 built from `amd/xdna-driver`
  (userspace only, `-nokmod`; in-tree `amdxdna` driver untouched). Reproducer
  (`env/aie-env.sh`, `run/run-m1.sh`), proof artifacts (run log, AIE2
  disassembly, `xrt-smi` enumeration, toolchain manifest), and full Void bring-up
  runbook (`INSTALL.md`).
- Phase 8 **M2**: single-core `matmul` (512×512×512 `i16→i32`, Peano, 32×32×32
  tiles) runs on the NPU at **92.2 GFLOPS** (NPU ~2.91 ms), output verified
  against a numpy reference — Peano GEMM bug #2793 did not affect this config.
  `run/run-m2.sh` + proof (`m2-matmul-run.txt`; `m2-kernel-aie2.disasm` shows the
  `vmac` accumulation loop on AIE2).
- Phase 8 **M3**: 1×1 **int8** `conv2d` (32×32, 64→64 ch, `kernels.conv2dk1_i8`,
  Peano) runs on one AIE core (~0.55 ms) and verifies against a PyTorch golden
  model — the int8 ML kernel relevant to YuNet's quantization. `run/run-m3.sh`
  + proof (`m3-conv2d-run.txt`, `m3-kernel-aie2.disasm`). Needs `torch` (CPU).
- Phase 8 **M4**: whole-array matmul across the full **4×4 = 16-tile** Phoenix
  array (`whole_array`, 512×512×512 `i16`, npu1_4col, Peano) at **891.5 GFLOPS**
  (NPU ~0.30 ms, ~9.7× single-core M2), numpy-verified — mlir-aie #1515
  (npu1_4col) did not bite this version. `run/run-m4.sh` + proof
  (`m4-whole-array-run.txt`, `m4-partition.txt`). **The M1–M4 ladder is complete.**
- Phase 8 stretch: **int8 matmul** (148 GFLOPS single-core / 979 GFLOPS 16-core,
  numpy-verified — Peano #2793/#2388 did not bite) and **fused int8 conv2d + ReLU**
  (uint8 out, torch-verified). Proof `stretch-int8-matmul.txt`, `stretch-conv-relu.txt`.
- Phase 8 stretch: **int8 matmul size sweep** (`run/sweep-int8.sh`, 16 tiles,
  512³–4096³, numpy-verified each point) — int8 throughput peaks **~2.0 TFLOPS at
  1024³** then declines (bandwidth-bound, well below the ~10 TOPS compute ceiling).
  Proof `stretch-int8-sweep.csv`.
- Phase 8 **M5 — real models on the NPU**: whole neural networks run end-to-end on
  the Phoenix array (Linux, open stack), each matching a CPU/torch golden
  reference: a **ResNet int8 bottleneck block** (~1.7 ms), **Google's Magika**
  file-type-detection NN — group0 (~0.6 ms, EVM −34.9 dB) and group2 (~0.4 ms,
  EVM −56.9 dB) — and a **ResNet `conv2_x` stage** (3 chained bottleneck blocks,
  ~2.4 ms). Lifts the M-ladder's "hand-written kernels only" scope. Proof
  `m5-models-run.txt` + `m5-magika-aie2.disasm` (xclbin `AIE_PARTITION` + AIE2
  `EM_AIE` core disassembly). See ADR-0009.
- ADR-0009: **AMD's official stack (VitisAI EP) is a verified dead end on
  XDNA1/Linux** as of 2026-06 (closed `voe` Linux module does not exist; Ryzen AI
  1.7.1 dropped XDNA1 from Linux support; RyzenAI-SW #341 open) — *not* a stale
  assumption (two adversarial reviewers, dated sources). Real models reach the NPU
  via the open stack instead.
- `xdna_probe.probe()`: new `npu-active-open-stack` verdict + `iron_runtime_importable`
  field — reports the open stack can drive the NPU (`pyxrt` + `xrt-smi` + device).
- Phase 8 **M6 — a 1B-parameter LLM on the NPU** (`project-c/m6-llm/`):
  **Llama-3.2-1B-Instruct** generates text with **every weight matmul** (q/k/v/o,
  SwiGLU gate/up/down, tied 128256-wide lm_head — 100% of params, >99% of FLOPs)
  executed on the XDNA1 NPU as a **bf16 GEMV with f32 accumulate**
  (`matvec_vectorized_bf16_f32`, one AIE2 core); CPU does only the parameter-free
  glue (RMSNorm, llama3 RoPE, GQA softmax, SiLU, argmax). Numerically faithful:
  next-token logits cosine **0.999992** vs full fp32, identical argmax/top-5;
  greedy output token-identical to the fp32 reference ("…is Paris. The capital of
  Germany"); a chat-templated instruction yields a fluent, EOS-terminated answer.
  No CPU-fallback path (opens `/dev/accel/accel0`, raises if absent). bf16 GEMV
  enabled by uncommenting the `mv.cc` bf16 combo + a `kernels.mv()` signature
  (`m6-llm/PATCH-bf16-gemv.md`) — int16 would overflow the int32 accumulator over
  K=8192. Honest scope: a *runs-and-is-correct* result, ~0.17 tok/s (single-core,
  per-call weight upload, M-blocked); the 780M iGPU still decodes this model ~2.2×
  faster. `run/run-m6.sh`; proof `m6-llm-run.txt`, `m6-chat-demo.txt`,
  `m6-gemv-bf16-shapes.txt`, `m6-gemv-bf16-aie2.disasm`, `m6-xrt-smi.txt`. See
  ADR-0010.
- M6 **interactive chat** (`llama_npu.py --interactive`, `run/chat-npu.sh`): a
  multi-turn streaming REPL on the NPU — Llama-3 chat template, KV cache carried
  across turns, token streaming, top-p/greedy sampling, `/reset` + `/exit`. The
  streaming decoder holds back incomplete multi-byte UTF-8 (emoji/CJK split
  across BPE tokens) until it resolves — fixing a delta-print corruption caught
  by an adversarial multi-reviewer pass. Proof `m6-chat-interactive.txt`.

### Changed
- ADR-0002's "no usable NPU compute on Linux" premise is **superseded** by
  ADR-0007 (open stack) — the NPU is now a compute target, not just a probe
  target. README narrative + status matrix updated (Phase 8 row).

### Removed
- `project-b/contrib/systemd/npu-status.{service,timer}` — the weekly NPU-status
  systemd units. They never fired on this Void/runit box, and the probe's purpose
  (detecting *when* the NPU became usable) is moot now that it is.
  `scripts/check-npu-status.fish` stays, runnable by hand.

### Fixed
- `scripts/check-npu-status.fish`: wrote to the dead
  `~/ObsidianVault/ClaudeMemory/...` path (gone after the Arch→Void migration) and
  shelled out to a bare `xrt-smi`. Now appends to repo-local
  `project-c/npu-status-log.md`, uses the open-stack `xrt-smi` under `/opt/xilinx/xrt`,
  and records the open-stack verdict (`npu-active-open-stack`) via
  `xdna_probe.probe()` + `project-c/npu_probe_verdict.py`.

### Docs
- ADR-0007 (mlir-aie/IRON/Peano over VitisAI for XDNA 1), ADR-0008
  (passthrough_kernel as M1; M1→M4 milestone ladder).

## [0.1.0] - 2026-04-19

First working twin-project build on the ThinkPad L16 Gen 2.

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
- Phase 4: concurrency harness (`bench/scripts/concurrency-run.fish`)
  runs the 20-prompt speculative bench twice — solo and corun with
  `presenced` polling the UVC webcam — captures `amdgpu_top -J`
  traces, `top -b` CPU samples, `/proc/meminfo` snapshots,
  `journalctl -k` windows, and a survival probe (kill `presenced`
  with llama up; stop llama with `presenced` up). Emits
  `bench/results/concurrency-<UTC>/verdict.json`. First run on
  ThinkPad L16 Gen 2: **max corun regression 0.82 % vs Phase 2
  baseline** (chat 0.82 / code 0.71 / factual 0.66 / reasoning
  0.70), both survival probes green, 0 OOM / 0 thermal throttle
  events — **Phase 4 acceptance gate passed**.
- Phase 5: daily-driver polish.
  - `project-b/contrib/hypr/presenced.conf` — Hyprland include with
    `SUPER+SHIFT+P` (toggle `presenced.service` via `systemctl --user`)
    and `SUPER+L` (manual `loginctl lock-session`). Uses the unit
    directly because the `presencectl` CLI was scoped out.
  - `project-b/contrib/waybar/{presenced.sh,config.jsonc,style.css}`
    — custom-module that reads FSM state from the journal (presenced
    does not export state externally); 2 s polling; left-click
    toggles, right-click follows logs in kitty.
  - `scripts/demo.fish` — non-destructive smoke gate: runs preflight,
    asserts both `--user` services active, fires one short
    `llama-speculative` prompt, reports presenced FSM state.
    Referenced from the README.
  - `docs/architecture.md` — ASCII dataflow diagram + short sections
    on why two services and what the NPU does (nothing, today).
  - Top-level `README.md` — replaces the Phase 0.5 stub. Covers what
    the project is, hardware tested, status matrix, install/configure
    for both projects and the Hyprland + waybar drop-ins, benchmark
    summary (Phase 2 and Phase 4), ADR list, Conventional Commits +
    no-AI-attribution rules.
  - **Phase 5 acceptance:** both services enable via
    `graphical-session.target`; `scripts/demo.fish` exits 0 on a
    freshly-logged-in session.
- Phase 6: XDNA 1 opportunistic probe.
  - `presenced --probe-npu` — deep probe that attempts an
    `onnxruntime.InferenceSession` with `VitisAIExecutionProvider`
    against a shipped 107-byte Identity ONNX model
    (`project-b/models/probe.onnx`, generated by
    `project-b/scripts/build-probe-onnx.py`). Emits one JSON line
    with `{ts, xrt_version, fw_version, device_present,
    providers_tried, providers_active, offloaded_ops, fallback_ops,
    verdict}`; exit 0 = NPU active, 2 = CPU fallback, 1 = EP
    unavailable / onnxruntime missing.
  - `scripts/check-npu-status.fish` — wraps `xrt-smi examine` +
    `presenced --probe-npu` into a single weekly run, appends a
    dated Markdown record to
    `~/ObsidianVault/ClaudeMemory/projects/npu-twin-npu-log.md`.
  - `project-b/contrib/systemd/npu-status.{service,timer}` — user
    units: `OnCalendar=weekly`, `Persistent=true`,
    `RandomizedDelaySec=1h`, `Nice=10`, `IOSchedulingClass=idle`.
  - `docs/xdna1-linux-status.md` — public pointer list of the
    upstream work that has to land before the probe can reach
    `verdict: npu-active` (RyzenAI-SW #341/#319/#350, llama.cpp
    #1499/#14377, FastFlowLM XDNA 1 exclusion).
  - **Phase 6 acceptance:** timer enabled and seeded; first run
    logged `verdict: onnxruntime-unavailable` (honest — no
    `python-onnxruntime` on the host today). The deliverable is the
    mechanism, not the verdict.
- Phase 7: v0.1.0 release.
  - `bench/results/acceptance-v0.1.0.md` — seven-gate acceptance
    matrix (baseline decode, speculative code, away FSM, concurrency,
    survival, XDNA probe mechanism, demo rc=0) and a four-row
    "deliberately skipped (opt-in, flag off)" table for gaze-focus /
    DND / brightness / unlock, called out as not-implemented rather
    than silently dropped.
  - `bench/results/demo-v0.1.0-20260419T120159Z.txt` — captured
    `scripts/demo.fish` output for the release.
  - README "Demo" section inlining that capture inside a
    `<details>` block, plus a short honesty note on the single-prompt
    accept rate and the per-class picture.

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

### Docs
- Migrated all NPU project notes from the legacy
  `~/ObsidianVault/NPU Projects/` folder into the shared Claude Memory
  vault at `~/ObsidianVault/ClaudeMemory/`, matching the conventions
  already in use there. ADRs 0001/0002/0003 mirrored into
  `ClaudeMemory/decisions/` (0004/0005/0006 already lived there);
  two new error notes captured in `ClaudeMemory/errors-and-fixes/`
  (`xrt-smi-mmap-memlock-eagain`, `render-group-not-in-session`);
  the project's hardware baseline, Project A/B detail, benchmarks
  tables, and commit log absorbed into
  `ClaudeMemory/projects/npu-twin.md`. Legacy folder retained read-only
  as a backup with an archive banner — no new writes there.

[Unreleased]: https://example.invalid/compare/v0.1.0...HEAD
[0.1.0]: https://example.invalid/releases/tag/v0.1.0
