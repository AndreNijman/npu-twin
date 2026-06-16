#!/usr/bin/env bash
# M2 proof: single-core matmul (C = A @ B) on the XDNA1 Phoenix NPU,
# verified against a numpy reference (assert_close_with_benchmark).
# Default: M=K=N=512, tile 32x32x32, i16 in / i32 out, Peano (use-chess=0).
set -uo pipefail
source ~/src/mlir-aie/aie-env314.sh
EX=~/src/mlir-aie/programming_examples/basic/matrix_multiplication/single_core
cd "$EX"
echo "=== xrt-smi device ==="
/opt/xilinx/xrt/bin/unwrapped/xrt-smi examine 2>/dev/null | grep -A2 "Device(s) Present" || true
echo "=== single-core matmul on NPU (JIT-compiles first time) ==="
timeout 900 python3 single_core.py "$@"
echo "RUN_RC=$?"
