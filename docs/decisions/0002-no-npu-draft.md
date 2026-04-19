# ADR-0002: No NPU-hosted draft model; both draft and target run on the 780M

Date: 2026-04-19
Status: Accepted

## Context

The original spirit of Project A was **heterogeneous speculative decoding**:
a small, fast draft model on the XDNA 1 NPU proposing tokens, a larger
target model on the iGPU verifying them. This is the native shape of
AMD's "OGA Hybrid" flow.

## Decision

Run **both** target and draft on the Radeon 780M via Vulkan. Do not
attempt to place the draft on the NPU.

## Reasoning

As of April 2026 the XDNA 1 Linux runtime story has no working LLM path:

- ONNX Runtime VitisAI Execution Provider silently falls back to CPU for
  all ops on XDNA 1 / Phoenix (RyzenAI-SW
  [#341](https://github.com/amd/RyzenAI-SW/issues/341),
  [#319](https://github.com/amd/RyzenAI-SW/issues/319),
  [#350](https://github.com/amd/RyzenAI-SW/issues/350)). "On the NPU"
  via VitisAI is a lie on Linux for this silicon.
- ONNX Runtime GenAI "OGA Hybrid" (NPU + iGPU LLM split) is **Windows-only
  and XDNA 2-only**. It does not run on this laptop.
- FastFlowLM (the Linux NPU backend in Lemonade Server) explicitly excludes
  XDNA 1: the 7000/8000/200-series chips are named as unsupported.
- `llama.cpp` has no merged or in-flight XDNA backend
  ([#1499](https://github.com/ggml-org/llama.cpp/issues/1499),
  [#14377](https://github.com/ggml-org/llama.cpp/issues/14377),
  [#10350](https://github.com/ggml-org/llama.cpp/discussions/10350)).

Trying to put the draft model "on the NPU" in any of these stacks would
actually put it on the CPU while lying about device placement. That would
be slower than running it on the iGPU next to the target, and it would
confuse anyone reading the code.

## Alternatives considered

- Run the draft on the CPU. Viable, but a 1B Q4_K_M on CPU pulls about the
  same generation throughput we would gain from speculation — it cancels.
- Wait for an XDNA 1 LLM runtime to land. Open-ended; do not block.

## Consequences

- A small piece of the story is lost: the build does not *actually* use
  the NPU for inference on XDNA 1. Phase 6 leaves a probe that detects
  this the moment it changes.
- On the 780M, draft+target sharing one device is normal for llama.cpp
  and works with `--device Vulkan0 -devd Vulkan0`.
- Memory pressure: target (8B Q4_K_M, ~4.9 GB) + draft (1B Q4_K_M,
  ~0.8 GB) + KV caches at 8K context fit comfortably in the 16 GB
  GTT slice the driver can carve out of 32 GB system RAM.
