# ADR-0008: passthrough_kernel as the M1 proof-of-concept; milestone ladder

Date: 2026-06-16
Status: Accepted

## Context

[ADR-0007](0007-mlir-aie-over-vitisai.md) adopts the open IRON/mlir-aie/Peano
stack. We need a first kernel that proves the toolchain + device round-trip
before investing in tuned compute.

## Decision

Use mlir-aie's `programming_examples/basic/passthrough_kernel` (IRON `@iron.jit`
flow) as **M1** — "data round-trips host → AIE core → host". Adopt this milestone
ladder for project-c:

| M | Target | Status |
|---|--------|--------|
| **M1** | passthrough_kernel runs on the NPU (`PASS!`) | ✅ done 2026-06-16 |
| **M2** | single-core `matmul` running + measured | next |
| **M3** | standalone int8 `conv2d` on one tile, vs CPU reference | stretch |
| M4 | whole-array (16-core) matmul | stretch |

## Reasoning

- passthrough needs **no** `xrt-smi` and **no** C++ host — the IRON
  `DefaultNPURuntime` opens the device via `pyxrt` and verifies with
  `assert_pass`. Smallest possible thing that exercises shim → driver → array
  → back. Ideal first-light.
- It is on `npu1_1col` by default — avoids the `npu1_4col` verification mismatch
  (mlir-aie #1515, "not planned"). Keep M1/M2 on 1 column.
- M2 (`matmul`) is the first *real* tuned-kernel result. Watch Peano GEMM
  correctness (mlir-aie #2793 / #2388, matmul-only); use current matched wheels
  and verify against a CPU reference before trusting numbers.

## Alternatives considered

- Start at `matmul` (M2) directly. Rejected: if it failed we couldn't tell a
  toolchain/device problem from a kernel-correctness problem. M1 isolates the
  former.
- `vector_scalar_add` C++ testbench. Equivalent proof but needs `xrt-smi` and a
  C++ XRT host; the IRON Python path is lower-friction for M1.

## Consequences

- M1's value is *learning/bring-up*, not throughput — a passthrough has none.
  Honest framing in `project-c/README.md`.
- Reproducible via `project-c/run/run-m1.sh`; proof artifacts under
  `project-c/proof/` (run log, AIE2 disassembly, xrt-smi enumeration, manifest).
- Next session starts at M2 against `basic/matrix_multiplication/single_core`.
