"""Parametric GEMV debug: {int16,bf16} x {scalar,vectorized} to isolate the
bf16 segfault. Run one combo at a time via argv to avoid one crash killing all.
"""
import argparse
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def build(dtype_name, vectorized):
    if dtype_name == "i16":
        din, dout = np.int16, np.int32
    else:
        din, dout = bfloat16, np.float32

    @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
    def design(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int],
               m: CompileTime[int] = 32, k: CompileTime[int] = 32):
        K_div_k = K // k
        M_div_m = M // m
        mvk = kernels.mv(dim_m=m, dim_k=k, input_dtype=din, output_dtype=dout,
                         vectorized=vectorized, use_chess=False)
        zk = mvk.zero
        A_ty = np.ndarray[(M, K), np.dtype[din]]
        B_ty = np.ndarray[(1, K), np.dtype[din]]
        C_ty = np.ndarray[(1, M), np.dtype[dout]]
        inA_ty = np.ndarray[(m, k), np.dtype[din]]
        inB_ty = np.ndarray[(k,), np.dtype[din]]
        outC_ty = np.ndarray[(m,), np.dtype[dout]]
        # vectorized kernel needs the 32-bit-word-transposed A layout; scalar
        # reads plain row-major.
        a_dims = [(m, 2), (k // 2, 2 * m), (2, 1)] if vectorized else None

        def core_fn(of_a, of_b, of_c, zero, mv):
            eo = of_c.acquire(1)
            zero(eo)
            for _ in range_(K_div_k):
                ea = of_a.acquire(1)
                eb = of_b.acquire(1)
                mv(ea, eb, eo)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

        Bf = ObjectFifo(inB_ty)
        Af = ObjectFifo(inA_ty, name="memA0")
        coreA = Af.cons().forward(dims_from_stream=a_dims)
        Cf = ObjectFifo(outC_ty, name="outC0")
        wk = Worker(core_fn, [coreA.cons(), Bf.cons(), Cf.prod(), zk, mvk])
        A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (M_div_m, K_div_k), prune_step=False)
        C_taps = TensorTiler2D.simple_tiler((1, M), (1, M), prune_step=False)
        b_tap = TensorTiler2D.simple_tiler((1, K), pattern_repeat=M_div_m, prune_step=False)[0]
        rt = Runtime()
        with rt.sequence(A_ty, B_ty, C_ty) as (a_in, b_in, c_out):
            rt.start(wk)
            rt.fill(Bf.prod(), b_in, b_tap)
            rt.fill(Af.prod(), a_in, A_taps[0])
            rt.drain(Cf.cons(), c_out, C_taps[0], wait=True)
        return Program(iron.get_current_device(), rt).resolve_program()

    return design, din, dout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["i16", "bf16"], required=True)
    ap.add_argument("--kernel", choices=["scalar", "vectorized"], required=True)
    ap.add_argument("-M", type=int, default=64)
    ap.add_argument("-K", type=int, default=64)
    args = ap.parse_args()
    vec = args.kernel == "vectorized"
    design, din, dout = build(args.dtype, vec)
    rng = np.random.default_rng(0)
    M, K = args.M, args.K
    if args.dtype == "i16":
        W = rng.integers(-50, 50, size=(M, K), dtype=np.int16)
        x = rng.integers(-50, 50, size=(K,), dtype=np.int16)
        ref = (W.astype(np.int64) @ x.astype(np.int64)).astype(np.int32)
    else:
        W = (rng.standard_normal((M, K)) * 0.05).astype(bfloat16)
        x = rng.standard_normal(K).astype(np.float32).astype(bfloat16)
        ref = W.astype(np.float32) @ x.astype(np.float32)
    A_t = iron.tensor(W.reshape(-1), dtype=din, device="npu")
    B_t = iron.tensor(x.reshape(-1), dtype=din, device="npu")
    C_t = iron.zeros(M, dtype=dout, device="npu")
    print(f"running {args.dtype}/{args.kernel} M={M} K={K} ...", flush=True)
    design(A_t, B_t, C_t, M=M, K=K, m=32, k=32)
    got = np.array(C_t.numpy()).reshape(M)
    rel = np.linalg.norm(got.astype(np.float64) - ref.astype(np.float64)) / (
        np.linalg.norm(ref.astype(np.float64)) + 1e-9)
    print("got[:6]", got[:6].tolist(), flush=True)
    print("ref[:6]", ref[:6].tolist(), flush=True)
    print(f"relL2={rel:.4e}  {'PASS' if rel < 0.03 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
