# npu-twin

Two independent services on one ThinkPad: speculative-decoding LLM
inference on the Radeon 780M iGPU, and a face-presence daemon that locks
Hyprland when you leave the chair. Built and measured on a **Lenovo
ThinkPad L16 Gen 2** (Ryzen 7 PRO 250 + Radeon 780M + XDNA 1 NPU) running
Arch Linux + Hyprland.

## What this is

A deliberately small reference build that answers two questions on
current AMD laptop silicon:

1. **Can you get a useful speedup from speculative decoding on a Vulkan
   iGPU without ROCm?** Yes — **1.59× on code, 1.61× on reasoning,
   1.42× overall** against the non-speculative baseline with the Llama
   3.1 8B + 3.2 1B Q4_K_M pair.
2. **Can a CPU-side webcam daemon share the box without stealing time
   from that inference?** Yes — **max 0.82 % regression** on the same
   suite while presenced polls the UVC webcam at 2 Hz; both services
   survive the other being killed; 0 OOM / 0 thermal-throttle events.

The other answer used to be the one that isn't in any marketing deck:
**through AMD's own stack, XDNA 1 has no usable Linux execution runtime** —
VitisAI silently falls back to CPU, OGA Hybrid and FastFlowLM are
Windows/XDNA2-only (ADR-0002). The device enumerates, `amdxdna` loads, XRT
lists it, and that's where the *AMD-stack* story ends for llama.cpp and OpenCV.

**Phase 8 (2026-06) changes the ending.** Using the fully open
IRON/mlir-aie/Peano stack — no Vitis, no VitisAI EP — a hand-written kernel now
runs on the Phoenix NPU under Linux (`PASS!`, ~125 µs on-NPU on Void/kernel-7.0).
See [`project-c/`](project-c/) and [ADR-0007](docs/decisions/0007-mlir-aie-over-vitisai.md).
The repo still publishes the AMD-stack gap honestly — it just no longer ends there.

## Hardware tested

- **Machine:** Lenovo ThinkPad L16 Gen 2 (21SCCTO1WW, BIOS R2UET27W)
- **CPU:** AMD Ryzen 7 PRO 250 (Zen 4, 8c/16t)
- **iGPU:** Radeon 780M (RDNA 3, `gfx1103`)
- **NPU:** AMD IPU `1022:1502` — XDNA 1 / Phoenix
- **RAM:** 32 GB
- **Webcam:** integrated UVC

Full stack table in [`docs/hardware.md`](docs/hardware.md).

## Status

| Phase | What | State |
|------:|------|:-----:|
| 0.5 | Repo bootstrap | ✅ |
| 1   | Driver + device preflight | ✅ |
| 2   | Project A: llama.cpp Vulkan speculative decoding | ✅ |
| 3   | Project B: `presenced` face-presence daemon | ✅ |
| 4   | Concurrency harness + coexistence gate | ✅ |
| 5   | Hyprland wiring + README + daily-driver polish | ✅ |
| 6   | Opportunistic XDNA 1 probe + honest gap writeup | ✅ |
| 7   | Demo + `v0.1.0` | ✅ |
| 8   | Project C: kernel on the XDNA 1 NPU via open mlir-aie/Peano (M1) | ✅ |

## Install

```fish
# 1. Clone
git clone https://github.com/AndreNijman/npu-twin ~/src/npu-twin
cd ~/src/npu-twin

# 2. Preflight — verifies kernel, NPU, RADV, render group, /dev/video0
./scripts/preflight.fish

# 3. Project A — llama.cpp + models + systemd --user unit
#    See project-a/README.md for the model-download + unit-install steps.

# 4. Project B — presenced (Python, YuNet ONNX)
cd project-b
python -m venv .venv && source .venv/bin/activate.fish
pip install -e .
./scripts/fetch-yunet.sh
systemctl --user link $PWD/contrib/systemd/presenced.service
systemctl --user daemon-reload
systemctl --user enable --now presenced.service

# 5. Smoke gate
cd ~/src/npu-twin && ./scripts/demo.fish
```

## Configure

### presenced

Environment variables (set via `systemctl --user edit presenced.service`
or a drop-in):

| Var | Default | Meaning |
|---|---|---|
| `PRESENCED_LOG_LEVEL` | `INFO` | Python logging level |
| `PRESENCED_FRAME_INTERVAL_S` | `1.0` | Poll period |
| `PRESENCED_GRACE_PERIOD_S` | `30.0` | Seconds of no-face before firing AWAY |
| `PRESENCED_AWAY_ACTION` | `dpms off` | `hyprctl dispatch <this>` on AWAY |
| `PRESENCED_PRESENT_ACTION` | `dpms on` | …on AWAY→PRESENT |
| `PRESENCED_DETECTOR` | `yunet` | Only `yunet` today (see ADR-0006) |
| `PRESENCED_YUNET_MODEL` | — | Path to YuNet ONNX (fetched by `fetch-yunet.sh`) |

Example: lock the session rather than DPMS-off:

```
Environment=PRESENCED_AWAY_ACTION=exec loginctl lock-session
Environment=PRESENCED_PRESENT_ACTION=true
```

### Hyprland

Drop in [`project-b/contrib/hypr/presenced.conf`](project-b/contrib/hypr/presenced.conf)
for `SUPER+SHIFT+P` (toggle presenced) and `SUPER+L` (manual lock):

```
source = ~/src/npu-twin/project-b/contrib/hypr/presenced.conf
```

### Waybar

Optional. See [`project-b/contrib/waybar/`](project-b/contrib/waybar/) —
status module that reads FSM state from the journal.

## Architecture

One ASCII diagram + dataflow notes: [`docs/architecture.md`](docs/architecture.md).

## Demo

`scripts/demo.fish` is the smoke gate: preflight, both `--user`
services active, one short `llama-speculative` prompt, presenced FSM
state. Captured on the tested ThinkPad L16 Gen 2 for the `v0.1.0`
release:

<details>
<summary>bench/results/demo-v0.1.0-20260419T120159Z.txt</summary>

```
== npu-twin demo ==
repo: /home/andre/src/npu-twin

[1/4] preflight
  [ok]   preflight clean (see /tmp/npu-twin-demo-preflight.log)

[2/4] --user services
  [ok]   llama-speculative.service active
  [ok]   presenced.service active

[3/4] llama-speculative one-shot
  [ok]   llama-speculative ran
         encoded   13 tokens in    0.435 seconds, speed:   29.876 t/s
         decoded   50 tokens in    9.487 seconds, speed:    5.270 t/s
         n_accept  = 30
         accept    = 19.737%

[4/4] presenced status
  [ok]   presenced active, last FSM state: present

== demo PASS ==
```

</details>

Two numbers to notice. The 5.27 t/s decode is low for this prompt
(non-code, non-speculative-friendly) — the Phase 2 bench gives the
real per-class picture (code: 13.45 t/s). The 19.7 % accept rate is
also low for a single short prompt; the Phase 2 code-class accept rate
is materially higher and documented in the benchmarks section and in
ADR-0003.

## Benchmarks

- **Phase 2 (Project A alone):** `bench/results/20260419T064428Z.json`
  — 20 prompts × 4 classes, greedy sampling, seed 42.
  code **13.45 t/s** · reasoning **13.71 t/s** · factual **10.63 t/s** ·
  chat **9.22 t/s**. Speedups vs non-speculative baseline:
  **code 1.59× / reasoning 1.61× / factual 1.25× / chat 1.08× (overall 1.42×)**.
- **Phase 4 (coexistence with presenced):**
  `bench/results/concurrency-20260419T101640Z/verdict.json`. Max corun
  regression **0.82 %** vs Phase 2. Survival probes green. 0 OOM /
  0 throttle events.

Reproduce:

```fish
./project-a/scripts/bench.fish                            # Phase 2
./bench/scripts/concurrency-run.fish                      # Phase 4
```

## Decisions

Every non-obvious choice has an ADR under [`docs/decisions/`](docs/decisions/):

- ADR-0001 — Vulkan/RADV over ROCm on gfx1103
- ADR-0002 — No NPU-hosted draft model; both LLMs on the 780M
- ADR-0003 — Llama 3.1 8B + 3.2 1B Q4_K_M as the target/draft pair
- ADR-0004 — YuNet ONNX over OpenCV Haar for `presenced`
- ADR-0005 — Revised Phase 3 CPU gate (<5 % system CPU, p95 <15 ms)
- ADR-0006 — Retire Haar after YuNet on-device validation

## Contributing

- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/).
  Small, atomic, one scope per commit.
- **No AI attribution** in commit messages, PR descriptions, or code
  comments. Write as yourself.
- **ADRs** for anything structural. Copy the existing format.
- Run `scripts/preflight.fish` and `scripts/demo.fish` before opening a PR.

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Vulkan backend,
  `llama-speculative`.
- [OpenCV](https://opencv.org/) + [YuNet](https://github.com/ShiqiYu/libfacedetection)
  — the face-detection frontend for presenced.
- The AMD upstream-Linux teams shipping `amdxdna` and XRT — the NPU
  enumerates and the kernel driver is in-tree, which is further than
  most accelerators get.
