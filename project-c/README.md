# project-c — running a kernel on the XDNA 1 NPU (open stack, no Vitis)

**Phase 8, milestone M1 reached (2026-06-16):** a hand-written kernel executes on
the ThinkPad L16's **AMD XDNA 1 "Phoenix" NPU** under Linux, compiled and run
entirely through the **open-source IRON / mlir-aie / Peano** toolchain — **no
AMD Vitis AI, no VitisAI ONNX Runtime EP**.

This directly overturns the premise the rest of this repo was built around
(`docs/decisions/0002-no-npu-draft.md`: *"XDNA 1 has no usable Linux execution
runtime"*). It does, now — just not AMD's. See
`docs/decisions/0007-mlir-aie-over-vitisai.md`.

## The result

`programming_examples/basic/passthrough_kernel` from mlir-aie, JIT-compiled to
the Phoenix array and run on the real device:

```
NPU time     (avg/min/max us): 125.4 / 102.1 / 135.7
End-to-end   (avg/min/max us): 215.9 / 191.2 / 238.1
PASS!
```

- `PASS!` ⇒ data round-tripped **host → AIE compute tile → host** bit-for-bit
  (`assert_pass(out, in)`); there is **no CPU-fallback path** in this flow, so a
  non-functional NPU hard-errors, it does not silently pass.
- `NPU time` is a hardware timestamp from the runtime, distinct from the Python
  end-to-end wall-clock — the work happened on the engine.

**M2 (2026-06-16):** a single-core `matmul` (512×512×512, `i16→i32`, Peano,
32×32×32 tiles) then ran at **92.2 GFLOPS** (NPU ~2.91 ms), output **verified
against numpy** (`assert_close_with_benchmark`) — a real `vmac` (vector
multiply-accumulate) loop on the AIE2 engine, not a passthrough.
`sg render -c 'bash run/run-m2.sh'`.

Proof artifacts are in [`proof/`](proof/):

| File | What it shows |
|------|---------------|
| `m1-passthrough-run.txt` | the `PASS!` run + NPU-side latency |
| `m1-kernel-aie2.disasm` | the kernel object is `ELF … Machine: EM_AIE (0x108)`, disassembling to **AIE2 VLIW** (`vldb/vlda/vst wh*`, `nopb;nopa;nops` bundles) — not x86 |
| `xrt-smi-examine.txt` | XRT enumerates `[0000:c6:00.1] RyzenAI-npu1  aie2  6x5` |
| `toolchain-manifest.txt` | exact versions used |
| `m2-matmul-run.txt` | M2 matmul `PASS!` + 92.2 GFLOPS, numpy-verified |
| `m2-kernel-aie2.disasm` | the matmul microkernel = a `vmac` accumulation loop on AIE2 |

## The stack

```
IRON (@iron.jit, Python)      structural design: Worker, ObjectFifo, Runtime
   │  emits MLIR
mlir-aie (aiecc)              lowering/placement/scheduling → xclbin + insts.bin
   │  compiles core C++ via
Peano (llvm-aie, LLVM 21)     open AIE2 LLVM backend — NO Vitis licence
   │  artifacts run via
pyxrt + XRT + libxrt_driver_xdna   userspace shim → in-tree amdxdna → /dev/accel/accel0
```

## Reproduce

```bash
# one-time toolchain bring-up (XRT-from-source + wheels):  see INSTALL.md
source project-c/env/aie-env.sh          # 3.14 IRON venv + system XRT + Peano + llvm-objcopy
sg render -c 'bash project-c/run/run-m1.sh'   # render group needed for /dev/accel/accel0
# expect: pyxrt+aie import OK, xrt-smi device table, then "PASS!"
```

## Hardware / versions (this run)

- ThinkPad L16 Gen 2 — Ryzen 7 PRO 250, **XDNA 1 "Phoenix"** NPU (`1022:1502`, aie2, 6×5 usable)
- Void Linux, kernel **7.0.11**, glibc 2.41
- in-tree `amdxdna` driver + firmware **`npu.sbin.1.5.5.391`** (protocol 7) — loads clean, no DKMS
- XRT **2.25.0** built from `amd/xdna-driver` (userspace only, `-nokmod`)
- `mlir_aie` **1.3.3.dev8+g0d49a88**, `llvm-aie` (Peano) **21.0.0.2026061601**, Python **3.14**

## Scope (honest)

M1 is a *passthrough* (toolchain + round-trip proof); **M2 is a real tuned
kernel** — a single-core `matmul` at ~92 GFLOPS, CPU-verified. The ladder (see
`docs/decisions/0008-passthrough-m1-poc.md`): M1 ✅, M2 ✅, **M3** = standalone
int8 `conv2d` (next), M4 = whole-array matmul. YuNet stays on CPU; a full model
on the NPU is explicitly out of scope.
