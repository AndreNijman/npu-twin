# ADR-0001: Use Vulkan/RADV over ROCm for llama.cpp on Radeon 780M

Date: 2026-04-19
Status: Accepted

## Context

Project A needs a stable GPU backend for `llama.cpp` on the integrated
Radeon 780M (RDNA 3, `gfx1103`) of the Ryzen 7 PRO 250. The two credible
options on Linux are:

1. **ROCm / HIP** via `llama.cpp-hip` with `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
2. **Vulkan / RADV** via `llama.cpp-vulkan`.

## Decision

Use **Vulkan/RADV**.

## Reasoning

- ROCm on `gfx1103` is unstable on current builds. `llama.cpp-hip` crashes
  on warmup on the 780M in recent ROCm releases
  (<https://github.com/ggml-org/llama.cpp/issues/20839>).
- RADV `PHOENIX` in Mesa 26.0.4 is mature and widely tested on 780M laptops.
- `llama.cpp` upstream treats Vulkan as a first-class backend; the
  speculative decoding path we need (`--model-draft`) works identically.
- Installing ROCm pulls tens of GB of runtime deps and introduces a second
  BLAS stack. Vulkan needs only Mesa (already present for display).
- Nothing in Project A is bottlenecked on ops that Vulkan lacks.

## Alternatives

- **Plan B1** (fallback): If Vulkan regresses on a future Mesa, try ROCm
  with the gfx override. Benchmark both; keep whichever is stabler and
  faster on this machine. Document the switch as a new ADR.
- **Plan C** (separate Windows writeup): If the user ever needs true
  NPU+iGPU hybrid LLM, that path lives on Windows with Ryzen AI SW +
  Lemonade Server OGA Hybrid. That is XDNA 2-centric and gains little on
  Hawk Point anyway.

## Consequences

- We depend on AUR's `llama.cpp-vulkan`. Expected; acceptable.
- No ROCm install. `rocm-*` packages stay off this machine.
- The `systemctl --user` unit forces `AMD_VULKAN_ICD=RADV` and binds
  `--device Vulkan0` so a later NVIDIA/secondary-GPU install does not
  silently shift device 0.
