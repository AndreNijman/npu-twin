# ADR-0009: AMD's official stack is a verified dead end on XDNA1/Linux; real models run on the NPU via the open stack instead

Date: 2026-06-16
Status: Accepted. Extends [ADR-0007](0007-mlir-aie-over-vitisai.md); lifts the
"a full model on the NPU is out of scope" line in
[ADR-0008](0008-passthrough-m1-poc.md).

## Context

After M1–M4 proved hand-written kernels run on Phoenix via the open stack, the
question became: can **AMD's official inference stack** (the VitisAI ONNX
Runtime Execution Provider / Ryzen AI Software) run a real *model* on this NPU
on Linux — the thing [ADR-0002](0002-no-npu-draft.md) said silently falls back
to CPU? And if not officially, can a real model run on the NPU **by any means**?

## Decision

1. Do **not** pursue AMD's official VitisAI EP on Linux for Phoenix. It is a
   verified dead end (below), not a stale assumption.
2. Run **real neural-network models** on the Phoenix NPU via the **open
   IRON / mlir-aie / Peano** stack (the same one M1–M4 used). This supersedes
   the M-ladder's "hand-written kernels only" scope.

## Reasoning — the official stack is dead on XDNA1/Linux (verified 2026-06)

A multi-angle investigation with dated sources and two adversarial reviewers
(one tasked to *refute* the "impossible" call) confirmed, current as of June
2026 — i.e. this is **not** the stale-April-2026 mistake ADR-0007 corrected:

- **AMD Ryzen AI Software 1.7.1 (2026-04-19) explicitly supports only Strix
  (STX) and Krackan (KRK) on Linux.** Phoenix/XDNA1 (npu1) and Hawk Point are
  *dropped* from official Linux support — a deliberate pivot to XDNA2.
  (`ryzenai.docs.amd.com/en/latest/linux.html`)
- **The closed-source `voe` runtime module has no Linux x86_64 build.** Without
  it the VitisAI EP loads but places **zero** ops on the NPU and silently runs
  on CPU. It is an AMD binary that does not exist for Linux/Phoenix — not
  something buildable or patchable from our side.
  (RyzenAI-SW [#341](https://github.com/amd/RyzenAI-SW/issues/341), open since
  2026-02-12, no AMD response.)
- FastFlowLM/Lemonade (the only working Linux NPU LLM path) **exclude XDNA1**;
  OGA Hybrid is Windows/XDNA2-only.
- On **Windows** Phoenix is officially (if "legacy-tier", manual `X1` target +
  `4x4.xclbin`) supported by Ryzen AI 1.7.1 — but this is a Linux project and
  the box runs Void; the Windows route is out of scope here.

So "make AMD's official stack work on Linux" is impossible for this silicon
today by any amount of effort on our side. The constraint that matters — run a
real model on the NPU on **Linux** — was then met a different way.

## Reasoning — real models DO run, via the open stack (M5)

mlir-aie ships whole-model examples that target npu1 (AIE2) and reuse our exact
working stack (XRT 2.25.0, in-tree amdxdna, mlir_aie 1.3.3.dev8, Peano
21.0.0.2026061601, Python 3.14). All four below ran on `[0000:c6:00.1]
RyzenAI-npu1` and **matched their CPU/torch golden reference** (proof in
`project-c/proof/m5-*`):

| Model | What it is | NPU time | Verify |
|---|---|---:|---|
| Bottleneck | ResNet int8 residual block (1×1→3×3→1×1 + skip) | ~1.7 ms | torch, atol |
| Magika group0 | Google's file-type-detection NN (bf16) | ~0.6 ms | EVM −34.9 dB |
| Magika group2 | second Magika sub-network (bf16) | ~0.4 ms | EVM −56.9 dB |
| ResNet conv2_x | three chained bottleneck blocks (a ResNet stage, int8) | ~2.4 ms | torch |

**Proof it is genuinely on-NPU, not CPU fallback:**
- Each harness opens `/dev/accel/accel0` via `pyxrt`/`DefaultNPURuntime` and
  **raises** if the device is absent — there is no silent CPU path in this flow.
- "NPU time" is an XRT hardware timestamp of 0.4–2.6 ms; the same conv workloads
  on CPU are tens-to-hundreds of ms.
- The model xclbin carries an `AIE_PARTITION` spanning Phoenix columns
  (`column_width 2`, `start_columns 1/2/3`); Peano emits per-tile core ELFs with
  machine `0x108` (`EM_AIE`) that disassemble to AIE2 VLIW (`movxm`, `paddb`,
  `nops;nopb;nopx;nopm`). See `project-c/proof/m5-magika-aie2.disasm`.

## Alternatives considered

- **VitisAI EP from source / Windows-`voe` transplant.** Dead: the partitioner
  is a closed Windows binary; PE-vs-ELF + the EP needing the Linux XRT/driver
  rules out Wine; no community port exists.
- **iree-amd-aie** (IREE's AIE backend, ONNX/torch → AIE). Viable but needs a
  multi-hour IREE+LLVM source build and its npu1 support is less proven than
  mlir-aie's; kept as a future "compiler-driven ONNX→NPU" path, not needed for
  this result.
- **MobileNet V3 e2e.** Designed for Strix's 4×8 array; the full network is
  larger than Phoenix's 16 compute tiles, so full e2e is not expected to place
  on npu1. Not pursued (would be a dishonest "push").

## Consequences

- The NPU is now a **model** compute target on Linux, not just a kernel one.
  The repo's strongest possible claim holds: real CNNs / a shipped Google model
  execute on XDNA1 under Linux, fully open, no Vitis.
- "ONNX model → NPU" via an automated frontend is still not wired up; these are
  IRON-described models compiled to the array (golden-verified), which satisfies
  "a real model runs on the NPU" but is not a drop-in `onnxruntime` provider.
- ADR-0002's "no NPU compute on Linux" premise is now doubly dead: not just
  kernels (ADR-0007) but **models**.
