#!/usr/bin/env bash
# int8 matmul GFLOPS sweep across square sizes on the XDNA1 Phoenix NPU.
# Whole-array (4x4 = 16 tiles, npu1_4col), i8 in / i32 out, 64x64x32 tiles, Peano.
# Each size is a separate @iron.jit compile (M/K/N are CompileTime). Verifies vs numpy.
set -uo pipefail
source ~/src/mlir-aie/aie-env314.sh
cd ~/src/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
# Valid square sizes for this config: M must make M/(m*n_aie_rows)=M/256 even,
# i.e. a multiple of 512 (256^3 / 768^3 fail the transfer-block row-count minimum).
SIZES="${*:-512 1024 1536 2048 4096}"
echo "size,dtype,cores,NPU_us_avg,GFLOPS,verify"
for S in $SIZES; do
  out=$(timeout 800 python3 whole_array.py --dtype_in i8 --dtype_out i32 \
        --n-aie-cols 4 -M "$S" -K "$S" -N "$S" -m 64 -k 64 -n 32 \
        --warmup 2 --iters 5 2>&1)
  g=$(printf '%s' "$out"  | grep -oE "NPU GFLOPS +: +[0-9.]+" | grep -oE "[0-9.]+$")
  us=$(printf '%s' "$out" | grep -oE "NPU time +\(avg/min/max us\): +[0-9.]+" | grep -oE "[0-9.]+$")
  if printf '%s' "$out" | grep -q "PASS!"; then v=PASS; else v=FAIL; fi
  echo "${S}x${S}x${S},i8->i32,16,${us:-NA},${g:-NA},$v"
done
echo "SWEEP_DONE"
