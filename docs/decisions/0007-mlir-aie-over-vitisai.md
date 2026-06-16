# ADR-0007: Use the open IRON/mlir-aie/Peano stack for XDNA 1, not VitisAI

Date: 2026-06-16
Status: Accepted. Supersedes the "no usable NPU compute on Linux" premise of
[ADR-0002](0002-no-npu-draft.md).

> Numbering note: an "ADR-0007: solvePnP gaze proxy" was drafted on the
> never-pushed `feat/gaze-estimator` (v0.2) branch, which did not survive the
> Arch→Void migration and never reached `main`. This repo's `main` never carried
> it, so 0007 is used here for the Phase 8 decision.

## Context

ADR-0002 (April 2026) concluded XDNA 1 / Phoenix had **no working Linux
execution runtime** and made the NPU a *probe target*, not a compute target —
because every path tried went through AMD's stack (VitisAI ONNX Runtime EP,
which silently falls back to CPU on Linux; OGA Hybrid, Windows/XDNA2-only;
FastFlowLM, XDNA2-only). That reasoning was correct *for AMD's stack*.

It missed a second path: the **fully open** toolchain — IRON (Python) → mlir-aie
(`aiecc`) → Peano (`llvm-aie`, the open LLVM AIE2 backend, no Vitis licence) →
XRT + the `amdxdna` driver. This compiles **hand-written kernels** (not ONNX
models) straight to the Phoenix array.

## Decision

Adopt the open IRON/mlir-aie/Peano stack as npu-twin's NPU execution path
(`project-c/`). Do **not** use VitisAI / ONNX Runtime for NPU offload.

## Reasoning

Demonstrated working on this exact box (2026-06-16, Void/kernel-7.0.11):
`passthrough_kernel` JIT-compiled to AIE2 and ran on `[0000:c6:00.1]
RyzenAI-npu1` → `PASS!`, NPU-side ~125 µs; the kernel object is `EM_AIE`
machine code (see `project-c/proof/`). No Vitis, no VitisAI EP, no CPU fallback.

Two findings made it tractable where ADR-0002 assumed a wall:
- The Arch-era firmware/driver **protocol mismatch is gone on kernel 7.0** — the
  in-tree `amdxdna` loads `npu.sbin.1.5.5.391` (protocol 7) cleanly. No DKMS.
- Only **userspace** must be built (XRT base + the `libxrt_driver_xdna` shim from
  `amd/xdna-driver`, `-nokmod`); the in-tree kernel driver is left untouched.

## Alternatives considered

- **Wait for VitisAI EP on Linux** (ADR-0002's implicit plan, tracked in
  `npu-twin-xdna1-upstream`). Still open (RyzenAI-SW #341). The open stack does
  not depend on it.
- **Build XRT's kernel module from xdna-driver** (`build.sh -release` + DKMS).
  Rejected: would `modprobe` a 0.15-era module that could demand newer firmware
  and **break the working in-tree 0.6 driver** (xdna-driver #1074/#1219). Use
  `-nokmod`.

## Consequences

- The NPU is now a **compute target**. The repo's headline gap closes: it
  *demonstrates* open-stack NPU execution rather than only documenting the
  VitisAI gap.
- This is hand-written kernels, **not** "ONNX model → NPU". A full YuNet on the
  NPU stays out of scope (see [ADR-0008](0008-passthrough-m1-poc.md)); YuNet
  remains on CPU in project-b.
- The weekly XDNA probe (`npu-twin-npu-log`) can honestly read `npu-active`: the
  probe is agnostic to *which* userspace drives the NPU, and an IRON kernel
  counts.
- Bring-up is non-trivial on a non-Ubuntu distro; the exact recipe is captured in
  `project-c/INSTALL.md`.
