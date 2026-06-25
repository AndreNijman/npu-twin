"""bf16 matrix-vector (GEMV) on the XDNA1 Phoenix NPU via IRON/mlir-aie/Peano.

Computes  C[M] = A[M,K] @ b[K]  in bf16 inputs with f32 accumulate, on one AIE2
compute core.  This is exactly an ``nn.Linear`` (out_features=M, in_features=K,
no bias): the HF weight tensor is stored [out, in] = [M, K], which is already the
``A`` matrix this kernel consumes — no transpose.

The bf16 path was enabled for npu-twin M6 by uncommenting the bf16 combo in
``mv.cc`` and teaching ``kernels.mv()`` the (bfloat16, float32) signature
(see PATCH-bf16-gemv.md).  Phoenix AIE2 does bf16 MAC with f32 accumulate
natively, so there is no int-overflow / quantization-scale problem (an int16
GEMV would overflow its int32 accumulator over K=8192).

Each distinct (M, K) compiles its own xclbin once; @iron.jit caches it on disk
and in-process, so the 5 Llama-1B shapes compile a handful of times total and
are reused across all 16 layers and every generated token.
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def _gemv_bf16(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    m: CompileTime[int] = 32,
    k: CompileTime[int] = 32,
):
    n_cores = 1
    M_div_m = M // m
    K_div_k = K // k

    # Vectorized bf16 GEMV: AIE2 bf16 MAC with f32 accumulate (products kept in
    # f32 -> more accurate than the scalar kernel, and faster). Enabled by
    # uncommenting the bf16 combo in mv.cc; see PATCH-bf16-gemv.md.
    matvec_kernel = kernels.mv(
        dim_m=m, dim_k=k, input_dtype=bfloat16, output_dtype=np.float32,
        vectorized=True, use_chess=False,
    )
    zero_kernel = matvec_kernel.zero

    dtype_in = bfloat16
    dtype_out = np.float32
    A_ty = np.ndarray[(M, K), np.dtype[dtype_in]]
    B_ty = np.ndarray[(1, K), np.dtype[dtype_in]]
    C_ty = np.ndarray[(1, M), np.dtype[dtype_out]]
    inA_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    inB_ty = np.ndarray[(k,), np.dtype[dtype_in]]
    outC_ty = np.ndarray[(m,), np.dtype[dtype_out]]

    # bf16 is 2 bytes, so the same "32-bit-word transposed" A layout the int16
    # path uses applies unchanged (transpose granularity = 2 elements).
    a_dims_from_stream = [(m, 2), (k // 2, 2 * m), (2, 1)]

    def core_fn(of_a, of_b, of_c, zero, matvec):
        elem_out = of_c.acquire(1)
        zero(elem_out)
        for _ in range_(K_div_k):
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matvec(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)

    B_fifo = ObjectFifo(inB_ty)
    a_fifo = ObjectFifo(inA_ty, name="memA0")
    coreA = a_fifo.cons().forward(dims_from_stream=a_dims_from_stream)
    outC = ObjectFifo(outC_ty, name="outC0")
    worker = Worker(
        core_fn,
        [coreA.cons(), B_fifo.cons(), outC.prod(), zero_kernel, matvec_kernel],
    )

    A_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (M_div_m, K_div_k), prune_step=False
    )
    C_taps = TensorTiler2D.simple_tiler((1, M), (1, M), prune_step=False)
    b_tap = TensorTiler2D.simple_tiler(
        (1, K), pattern_repeat=M_div_m, prune_step=False
    )[0]

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a_in, b_in, c_out):
        rt.start(worker)
        rt.fill(B_fifo.prod(), b_in, b_tap)
        rt.fill(a_fifo.prod(), a_in, A_taps[0])
        rt.drain(outC.cons(), c_out, C_taps[0], wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


# in-process cache of device tensors keyed by shape is unnecessary; @iron.jit
# already caches the compiled program. We just allocate per call.
# The B-vector fill uses pattern_repeat = M_block/m, which maps to a hardware
# DMA buffer-descriptor wrap dimension capped at 64. With m=32 that limits one
# kernel invocation to M_block <= 64*32 = 2048 rows. Large-M Linears (gate/up
# 8192, lm_head 128256) are split into <=2048-row blocks on the host and
# concatenated. K is unconstrained (down_proj K=8192 compiles fine).
MAX_M_BLOCK = 2048


def gemv(W_bf16: np.ndarray, x_bf16: np.ndarray, m: int = 32, k: int = 32) -> np.ndarray:
    """C[M] = W[M,K] @ x[K] on the NPU. W, x are bfloat16; returns float32[M]."""
    M, K = W_bf16.shape
    assert x_bf16.shape == (K,), f"x must be [{K}], got {x_bf16.shape}"
    assert M % m == 0 and K % k == 0, f"({M},{K}) not divisible by tile ({m},{k})"
    # activation is identical across all M-blocks -> upload once
    B_t = iron.tensor(np.ascontiguousarray(x_bf16).reshape(-1), dtype=bfloat16, device="npu")
    out = np.empty(M, dtype=np.float32)
    start = 0
    while start < M:
        mb = min(MAX_M_BLOCK, M - start)
        Wb = np.ascontiguousarray(W_bf16[start:start + mb])
        A_t = iron.tensor(Wb.reshape(-1), dtype=bfloat16, device="npu")
        C_t = iron.zeros(mb, dtype=np.float32, device="npu")
        _gemv_bf16(A_t, B_t, C_t, M=mb, K=K, m=m, k=k)
        # copy the device-backed view into the output (avoid use-after-free)
        out[start:start + mb] = C_t.numpy().reshape(mb)
        start += mb
    return out


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    shapes = [(256, 256), (2048, 2048), (512, 2048), (8192, 2048), (2048, 8192)]
    if "--full" in sys.argv:
        shapes.append((128256, 2048))  # lm_head
    all_ok = True
    for (M, K) in shapes:
        W = rng.standard_normal((M, K)).astype(np.float32) * 0.05
        x = rng.standard_normal(K).astype(np.float32)
        Wb = W.astype(bfloat16)
        xb = x.astype(bfloat16)
        ref = (Wb.astype(np.float32) @ xb.astype(np.float32))
        got = gemv(Wb, xb)
        # bf16 relative error metric
        rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9)
        cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-9))
        ok = rel < 0.03 and cos > 0.999
        all_ok &= ok
        print(f"({M:>6},{K:>5})  relL2={rel:.4e}  cos={cos:.6f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
