# Enabling bf16 matrix-vector (GEMV) in mlir-aie for M6

The XDNA1 LLM (M6) runs every weight matmul as a **bf16 GEMV with f32
accumulate** on one AIE2 core. The mlir-aie `mv` (matrix-vector) kernel already
*supports* bf16 in C++, but ships with the bf16 instantiation commented out and
the Python `kernels.mv()` wrapper hard-restricted to `int16`. Two tiny edits in
the installed `mlir_aie` package enable it. They are reproduced here because the
package lives in the venv (`~/src/mlir-aie/ironenv314/.../site-packages/mlir_aie`)
and is not under version control.

Why bf16 and not int16: an `int16 × int16` GEMV summed over K=8192 (down_proj /
the residual width) **overflows the int32 accumulator** (8192 × ~1e9 ≫ 2^31),
which would force an ~8-bit effective range — too lossy for a coherent 1B model.
bf16 inputs with an **f32 accumulator** have no overflow and are the model's
native dtype. AIE2 (Phoenix) does bf16 MAC with f32 accumulate in hardware
(the same capability M5's Magika run used).

## Edit 1 — `include/aie_kernels/aie2/mv.cc`

Uncomment the bf16 combo so the kernel emits `matvec_vectorized_bf16_f32`,
`matvec_scalar_bf16_f32`, and `zero_vectorized_f32` / `zero_scalar_f32`:

```diff
 #define combos(X)                                                              \
-  /* X(bfloat16, bf16, float, f32, accfloat) */                                \
+  X(bfloat16, bf16, float, f32, accfloat)                                      \
   X(int16, i16, int32, i32, acc32)
```

The `matvec_vectorized<>` template already `static_assert`s bf16 as a permitted
`T_in`; uncommenting just instantiates it. The two combos produce
differently-mangled C symbols, so this only *adds* the bf16 symbols (the int16
path is untouched).

## Edit 2 — `python/aie/iron/kernels/linalg.py`, function `mv()`

Allow `(bfloat16, np.float32)` alongside `(np.int16, np.int32)` and pick the
symbol suffix from the dtype pair instead of hard-coding `i16_i32`:

```python
st_in = np.dtype(input_dtype).type
st_out = np.dtype(output_dtype).type
_suffixes = {
    (np.int16, np.int32): ("i16", "i32"),
    (bfloat16, np.float32): ("bf16", "f32"),
}
if (st_in, st_out) not in _suffixes:
    raise ValueError(...)
in_suf, out_suf = _suffixes[(st_in, st_out)]
# ... a_ty/b_ty use np.dtype[st_in]; c_ty uses np.dtype[st_out];
# extern name f"{prefix}_{in_suf}_{out_suf}"; zero f"{zero_prefix}_{out_suf}"
```

(`bfloat16` is already imported at the top of `linalg.py`.)

## Result

`kernels.mv(..., input_dtype=bfloat16, output_dtype=np.float32, vectorized=True)`
now compiles and runs. Verified vs numpy across every Llama-1B shape — relL2
~1e-7…1e-6, cos=1.000000 (see `proof/m6-gemv-bf16-shapes.txt`):

| (M, K) | what | relL2 |
|--------|------|------:|
| (2048, 2048) | q_proj / o_proj | 4.0e-7 |
| (512, 2048)  | k_proj / v_proj | 4.6e-7 |
| (8192, 2048) | gate_proj / up_proj | 4.0e-7 |
| (2048, 8192) | down_proj | 9.8e-7 |
| (128256, 2048) | tied lm_head | 4.1e-7 |

## Two gotchas found while wiring this up

1. **Use vectorized, not scalar.** The *vectorized* bf16 kernel keeps products
   in f32 inside `aie::accumulate` (relL2 ~1e-7) and is faster; the scalar
   kernel rounds each product to bf16 (relL2 ~1.5e-2). Both are correct enough,
   vectorized is strictly better.
2. **DMA buffer-descriptor wrap limit ⇒ M-blocking.** The B-vector fill repeats
   the activation `M/m` times via one DMA BD whose wrap dimension is capped at
   64. With m=32 that caps a single kernel call at `M_block ≤ 2048` rows. Larger
   Linears (gate/up 8192, lm_head 128256) are split into ≤2048-row blocks on the
   host and concatenated (see `npu_gemv.py`). K is unconstrained (down_proj
   K=8192 compiles directly).

Both edits are candidates for an upstream mlir-aie PR ("enable bf16 GEMV"),
consistent with npu-twin's earlier contributions (amd/xdna-driver#1424,
Xilinx/mlir-aie#3175).
