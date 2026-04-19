# npu-twin — architecture

Two independent daemons share one Ryzen 7 PRO 250 + Radeon 780M + XDNA 1
NPU on the ThinkPad L16 Gen 2. Both are `systemctl --user` units; neither
knows about the other at runtime. The concurrency harness measures how
well they coexist.

```
                         ┌─────────────────────────────────────┐
                         │       Arch Linux / Hyprland         │
                         │  (user session, graphical-session)  │
                         └──────────────┬──────────────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              │                                                   │
   ┌──────────▼────────────┐                        ┌─────────────▼────────────┐
   │  Project A            │                        │  Project B               │
   │  llama-speculative    │                        │  presenced               │
   │  (--user svc)         │                        │  (--user svc)            │
   ├───────────────────────┤                        ├──────────────────────────┤
   │  target: Llama 3.1 8B │                        │  capture: /dev/video0    │
   │  draft:  Llama 3.2 1B │                        │    (OpenCV + GStreamer)  │
   │  Q4_K_M, RADV/Vulkan0 │                        │  detect: YuNet INT8 ONNX │
   │  --draft-max 8        │                        │    (cv2.FaceDetectorYN)  │
   │  --draft-min 2        │                        │  FSM: present / grace /  │
   │  --draft-p-min 0.6    │                        │       away  (configurable│
   │  seed 42, temp 0      │                        │       grace window)      │
   └──────────┬────────────┘                        └──────────────┬───────────┘
              │                                                    │
              │ HTTP  :11434                                       │ hyprctl dispatch
              │ /completion                                        │ (HYPRLAND_INSTANCE_
              ▼                                                    │  SIGNATURE guard)
   ┌───────────────────────┐                        ┌──────────────▼───────────┐
   │  bench.fish /         │                        │  Hyprland IPC            │
   │  concurrency-run.fish │                        │  — dpms off / lock /     │
   │  (20-prompt suite)    │                        │    user-supplied action  │
   └───────────┬───────────┘                        └──────────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │  bench/results/*.json │
   │  verdict.json per run │
   └───────────────────────┘

                       ┌──────────────────────────────┐
                       │  Radeon 780M iGPU (gfx1103)  │  ← both paths hit
                       │  RADV/Vulkan                 │    this one device;
                       │  (Mesa 26+)                  │    concurrency probe
                       └──────────────────────────────┘    proves <10 % tok/s
                                                           regression on code
                       ┌──────────────────────────────┐
                       │  XDNA 1 NPU (Phoenix)        │  ← present + enumerated;
                       │  /dev/accel/accel0           │    no Linux runtime for
                       │  amdxdna 0.6.0               │    llama.cpp or OpenCV
                       │  XRT 2.21.75                 │    today. presenced
                       │                              │    runs an opportunistic
                       │                              │    xdna_probe at start,
                       │                              │    status-only. See
                       │                              │    ADR-0002.
                       └──────────────────────────────┘
```

## Data flow

**Project A (inference).** `llama-speculative` accepts a prompt, runs the
draft model for up to 8 tokens, verifies with the target model in one
forward pass, commits accepted prefix, iterates. All on Vulkan0 (Radeon
780M). Output returned on stdout for CLI invocations, or via the
OpenAI-style HTTP endpoint the unit exposes on `127.0.0.1:11434`.

**Project B (presence).** `presenced` polls the webcam at `FRAME_INTERVAL_S`
(default 1.0 s). Each frame goes through YuNet; the FSM decides
`present` / `away_grace` / `away` based on detection + `GRACE_PERIOD_S`
(default 30 s). On `-> away` the `PRESENCED_AWAY_ACTION` fires; on
`away -> present` the `PRESENCED_PRESENT_ACTION` fires. Dispatch is
`hyprctl dispatch <action>`, gated on a valid `HYPRLAND_INSTANCE_SIGNATURE`
in the environment (so the daemon no-ops cleanly outside Hyprland).

## Why two services, not one

They have nothing to say to each other. Inference doesn't care who's in
front of the webcam; presence doesn't care whether a draft was accepted.
Separate unit files make each restartable, observable, and benchable in
isolation. The concurrency harness (`bench/scripts/concurrency-run.fish`)
runs Phase 2's 20-prompt suite twice — once with presenced stopped, once
with it polling — and writes `verdict.json` with per-class regression vs
the Phase 2 baseline and a survival probe (kill each with the other up).

## What the NPU does

Today: nothing at runtime. The XDNA 1 upstream Linux story — amdxdna in
kernel, XRT CLI in userspace, no Vitis AI execution provider on Linux —
means neither llama.cpp nor OpenCV's DNN backend can target it. ADR-0002
records the decision to ship both models on the 780M and leave the NPU
for an opportunistic probe. `project-b/presenced/xdna_probe.py` runs at
presenced startup and logs device + XRT + Vitis EP status; it does not
offload. The repo's value here is documenting that gap honestly on real
hardware.

## Files to read first

| File | What |
|---|---|
| `docs/hardware.md` | The tested L16 Gen 2 stack — kernel, amdxdna, firmware, XRT, Mesa versions |
| `docs/decisions/` | ADR-0001 … ADR-0006 — why Vulkan over ROCm, why no NPU draft, etc. |
| `project-a/scripts/bench.fish` | 20-prompt speculative-decoding bench |
| `project-b/presenced/__main__.py` | Main loop — detector → FSM → hyprctl |
| `bench/scripts/concurrency-run.fish` | Phase 4 concurrency + survival harness |
| `scripts/demo.fish` | One-shot smoke gate that exercises both services |
