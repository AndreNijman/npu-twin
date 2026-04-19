# Hardware and Driver Notes

This repo targets the **Lenovo ThinkPad L16 Gen 2** (model 21SCCTO1WW, BIOS
R2UET27W). Other Ryzen 7040/7045/8040/200-series "Hawk Point" laptops should
also work — the same CPU/iGPU/NPU silicon is shared.

## Silicon

- **CPU:** AMD Ryzen 7 PRO 250 (Hawk Point refresh, Zen 4, 16 threads)
- **iGPU:** Radeon 780M (RDNA 3, `gfx1103`)
- **NPU:** AMD IPU Device `1022:1502` — XDNA 1 / NPU Phoenix generation

## Linux stack (as tested)

| Component | Version |
|-----------|---------|
| Kernel | `6.19.10-arch1-1` (requires ≥ 6.14 for `amdxdna`) |
| amdxdna driver | `0.6.0` (in-tree) |
| NPU firmware | `1.5.2.380` |
| XRT | `2.21.75-6` (Arch `extra`) |
| XRT plugin | `xrt-plugin-amdxdna` `2.21.75-2` (Arch `extra`) |
| Mesa | `26.0.4-arch1.1` |
| Vulkan API | `1.4.335` via RADV `PHOENIX` |

## Device nodes

- `/dev/accel/accel0` — NPU, owner `root:render`, mode `0666`
- `/dev/dri/renderD128` — 780M render node, owner `root:render`

Both require the user to be in the `render` group.

## Required runtime configuration

1. `* soft memlock unlimited` / `* hard memlock unlimited` in
   `/etc/security/limits.d/99-amdxdna.conf`. XRT mmaps >64 MB for NPU
   workloads and the default `ulimit -l` (8192 KB) is too small. Takes
   effect on next PAM login — not on `newgrp`/`sg`.
2. `sudo usermod -aG render,video $USER` — then **log out and back in**.
3. Kernel cmdline must not contain `amd_iommu=off`; XDNA uses IOMMU SVA.

After both, `xrt-smi examine` as an unprivileged user enumerates
`RyzenAI-npu1` at `[0000:c6:00.1]`.

## Reality check (April 2026)

The hardware is healthy on Linux. The **runtime story** is not:

- No functioning Linux LLM runtime exists for XDNA 1 today. VitisAI EP falls
  back to CPU (RyzenAI-SW #341, #319, #350), ONNX Runtime GenAI "Hybrid" is
  Windows+XDNA 2 only, FastFlowLM excludes XDNA 1 by design, llama.cpp has
  no merged XDNA backend (llama.cpp #1499, #14377).
- ROCm on `gfx1103` is unstable on recent builds; llama.cpp `-hip` crashes
  on warmup on the 780M (llama.cpp #20839).

**Therefore:**
- Project A runs LLMs on the 780M via **Vulkan/RADV** (not ROCm, not NPU).
- Project B runs on **CPU** (ONNX Runtime CPUExecutionProvider). A
  `use_npu` feature flag is preserved so the switch is cheap when AMD's
  Linux stack matures.

See `docs/decisions/` for the ADRs backing these choices.
