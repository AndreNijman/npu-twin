# Project A — Speculative decoding on Radeon 780M (Vulkan)

Serves an OpenAI-compatible endpoint on `127.0.0.1:11434` backed by
`llama.cpp` with **draft-model speculative decoding**. Both the target
(Llama 3.1 8B Q4_K_M) and the draft (Llama 3.2 1B Q4_K_M) run on the
iGPU via `llama.cpp-vulkan` (RADV PHOENIX).

## Why Vulkan, not ROCm, not NPU

On Ryzen 7 PRO 250 (XDNA 1, `gfx1103`) in April 2026:

- ROCm on `gfx1103` is unstable. `llama.cpp-hip` crashes on warmup on
  the 780M on recent ROCm builds (upstream llama.cpp #20839).
- There is no working Linux LLM runtime for XDNA 1 NPUs (see
  `docs/hardware.md` and `docs/decisions/`).
- Vulkan/RADV on `gfx1103` is both faster and stable on this machine.

## Dependency

- `llama.cpp-vulkan` from Arch AUR (e.g. `paru -S llama.cpp-vulkan`).
  Tested at build `b8840`. Provides `llama-server`, `llama-cli`,
  `llama-bench`.
- `python-huggingface-hub` (provides `hf` CLI) for model fetches.

## Run

```fish
./scripts/fetch-models.fish       # downloads GGUFs (gitignored)
./scripts/smoke.fish              # quick standalone generation
systemctl --user enable --now llama-speculative.service
curl http://127.0.0.1:11434/v1/models
```

## Benchmarks

See `./scripts/bench.fish` and `bench/results/`.
