# XDNA 1 Linux — userspace runtime status

**Short version:** on a `gfx1103` / Phoenix laptop in 2026-04, the NPU
enumerates and the kernel driver works, but no user-space runtime can
actually put a PyTorch/ONNX op on the device. This repo's weekly probe
([`scripts/check-npu-status.fish`](../scripts/check-npu-status.fish))
logs that state so it becomes obvious when the gap closes.

This note is a pointer list. Check it if the probe log shows a verdict
change, or before proposing an NPU-offload PR.

## What works today

- Kernel driver `amdxdna` 0.6.0, in-tree since 6.14.
- `/dev/accel/accel0` enumerates, owned by `root:render`.
- XRT 2.21.75 + `xrt-plugin-amdxdna` 2.21.75 (Arch `extra`).
- NPU firmware 1.5.2.380.
- `xrt-smi examine` reports the device (may need `LimitMEMLOCK=infinity`
  — see `presenced.service`).

## What doesn't work

- `onnxruntime` with VitisAIExecutionProvider: no installable Arch
  package; PyPI wheels don't bundle the EP.
- Even with a built-from-source EP, VitisAI on XDNA 1 Linux silently
  falls back to CPU for every op today.
- llama.cpp has no NPU backend.

## Upstream issues tracked

| Project | Issue | What it unblocks |
|---|---|---|
| AMD RyzenAI-SW | [#341](https://github.com/amd/RyzenAI-SW/issues/341) | VitisAI EP CPU fallback on Linux — primary blocker |
| AMD RyzenAI-SW | [#319](https://github.com/amd/RyzenAI-SW/issues/319) | PHX operator-set regressions |
| AMD RyzenAI-SW | [#350](https://github.com/amd/RyzenAI-SW/issues/350) | HPT regressions (tracks XDNA 2 progress vs XDNA 1) |
| llama.cpp | [#1499](https://github.com/ggerganov/llama.cpp/issues/1499) | Generic AMD NPU backend feature request |
| llama.cpp | [#14377](https://github.com/ggerganov/llama.cpp/issues/14377) | More recent XDNA-specific discussion |

Third-party runtimes worth watching: FastFlowLM (Lemonade) explicitly
excludes XDNA 1 / Phoenix from supported NPUs — a useful signal for
where community tooling expects the ecosystem to stabilise.

## When this page is out of date

- If `scripts/check-npu-status.fish` emits a verdict other than
  `onnxruntime-unavailable`, revisit this list.
- If `docs/decisions/0002-no-npu-draft.md` (ADR-0002) gets superseded,
  update both docs together.

The private ops-side mirror of this note, with dated re-checks, lives
in the author's Obsidian vault at
`ClaudeMemory/projects/npu-twin-xdna1-upstream.md`.
