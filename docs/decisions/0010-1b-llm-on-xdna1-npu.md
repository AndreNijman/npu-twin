# ADR-0010: A 1B-parameter LLM runs on the XDNA1 NPU (Llama-3.2-1B via bf16 GEMV, open stack)

Date: 2026-06-25
Status: Accepted. Extends [ADR-0009](0009-official-stack-dead-real-models-via-open.md)
(real models via the open stack) to a full autoregressive LLM. Closes the
"assembled LLM = frontier" gap noted in the 2026-06-25 research
([best-llm-on-xdna1-npu]). ADR-0002's premise is now triply dead.

## Context

M1–M4 put hand-written kernels on the Phoenix NPU; M5 put whole CNNs (ResNet
blocks, Google's Magika) on it. The open question the project kept flagging: can
a **real large language model** — the thing AMD's dropped Linux stack can't run
on XDNA1 — execute on this NPU at all?

The 2026-06 research found **no turnkey LLM runtime exists for Phoenix NPU1 on
Linux** (FastFlowLM/Lemonade/RyzenAI-SW are XDNA2 and/or Windows; ADR-0009).
mlir-aie ships the transformer *building blocks* (matmul, GEMV, rope, softmax,
norm, swiglu) targeting AIE2, but **no assembled LLM**. It also concluded the
best NPU-runnable target, if built, is a **~1B model** (Llama 3.2 1B).

## Decision

Assemble **Llama-3.2-1B-Instruct** so that **every weight matmul runs on the
XDNA1 NPU** via the open IRON/mlir-aie/Peano stack:

- The 7 Linears per layer (q/k/v/o projections, SwiGLU gate/up/down) and the
  tied **lm_head** (128256×2048) are each dispatched to the AIE2 array as a
  **bf16 matrix-vector multiply with f32 accumulate** (`matvec_vectorized_bf16_f32`,
  one compute core). That is **100% of the model's 1.24B parameters** and
  **>99% of its FLOPs**.
- The parameter-free glue — RMSNorm, llama3 RoPE, GQA attention scores/softmax,
  SiLU, residual adds, argmax — runs on the CPU in float32. (mlir-aie *has*
  NPU kernels for norm/rope/softmax/swiglu; wiring those onto the array too is
  future work, not needed to satisfy "the LLM runs on the NPU".)

bf16, not int: an `int16 × int16` GEMV summed over K=8192 overflows the int32
accumulator (≫ 2³¹), which would cap useful range at ~8 bits — too lossy for a
coherent 1B model. bf16 inputs with an **f32 accumulator** are the model's native
dtype and overflow-free; AIE2 does bf16 MAC in hardware (M5's Magika used bf16).
The mv kernel's bf16 combo ships commented out in `mv.cc`; we enabled it
(`m6-llm/PATCH-bf16-gemv.md`).

## Proof it runs on the NPU, correctly

- **No CPU fallback.** Each GEMV opens `/dev/accel/accel0` via pyxrt and raises
  if the device is absent; the kernel object is `EM_AIE` (0x108) AIE2 VLIW
  (`m6-llm/`/`proof/m6-gemv-bf16-aie2.disasm`).
- **Numerically faithful.** Prefilling "The capital of France is" and comparing
  the next-token logits, NPU-forward vs full-fp32-forward:
  cosine **0.999992**, relL2 3.9e-3, **identical argmax** (`12366 = " Paris"`)
  and **identical top-5** (`[12366, 264, 539, 1131, 1101]`).
- **Coherent generation.** Greedy decode on the NPU continues
  "The capital of France is" → "**Paris. The capital of Germany**" — token-for-token
  identical to the fp32 reference trajectory. Proof `proof/m6-llm-run.txt`.
- Per-shape GEMV vs numpy: relL2 ~1e-7…1e-6, cos=1.000000 across all six Llama
  shapes incl. the 128256 lm_head (`proof/m6-gemv-bf16-shapes.txt`).

## Honest scope

- This is a **"it runs and is correct"** result, **not a speed result.**
  ~0.17 tok/s: single AIE compute core, the weight is re-uploaded host→device
  every call, and ~255 separate kernel invocations per token (the M-blocking
  below + 16 layers + lm_head). The research already established decode is
  DRAM-bandwidth-bound and the **780M iGPU is ~2.2× faster** on this same model
  (44 tok/s); the NPU's only real LLM wins are **prefill/TTFT** and **perf/watt**.
  Speeding M6 up (resident weights, multi-core/whole-array GEMV, on-NPU norm/rope)
  is future work.
- **M-blocking.** The B-vector DMA repeats the activation `M/m` times through one
  buffer-descriptor wrap dimension capped at 64 ⇒ a single kernel call is limited
  to `M ≤ 2048` rows; large Linears (gate/up 8192, lm_head 128256) are split into
  ≤2048-row blocks on the host and concatenated. K is unconstrained.
- The model is loaded from HF `unsloth/Llama-3.2-1B-Instruct` (bf16 safetensors);
  these are IRON-driven GEMV calls, not an automated ONNX/GGUF→NPU frontend.

## Alternatives considered

- **int8/int16 quantized matmul (the proven M2/M4 kernels).** Rejected for the
  default: int32-accumulator overflow over K=8192 forces ~8-bit range and
  activation-outlier handling; bf16 is simpler and the model's native format.
- **iree-amd-aie (compiler-driven ONNX→AIE).** Still the future "drop-in" path;
  needs a multi-hour IREE+LLVM build and its npu1 support is less proven than
  mlir-aie's. Not needed for this result (ADR-0009 alternatives still hold).
- **Whole-array (16-tile) GEMV.** Would cut latency ~10× but the shipped
  `matrix_vector` is single-core; multi-core GEMV is a future optimization.

## Consequences

- The NPU is now an **LLM** compute target on Linux, not just a kernel (ADR-0007)
  or CNN (ADR-0009) one. The repo's strongest claim: **a real 1B-parameter
  language model executes on AMD XDNA1 under Linux, fully open, no Vitis** — the
  exact silicon AMD dropped from its Linux stack.
- ADR-0002 ("no NPU compute on Linux") is now dead three ways over: kernels,
  models, and a full LLM.
- The M-ladder gains **M6**: M1 passthrough · M2 matmul · M3 int8 conv2d ·
  M4 16-core matmul · M5 real CNNs · **M6 a 1B LLM**.
