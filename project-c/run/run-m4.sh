#!/usr/bin/env bash
# M4 proof: whole-array (4x4 = 16-core) matmul C = A @ B on the XDNA1 Phoenix NPU,
# verified against a numpy reference. Default: M=K=N=512, m/k/n=64/64/32,
# --n-aie-cols 4 (npu1_4col, 16 compute tiles), i16 in/out, Peano (no Vitis).
# Pass e.g. `--n-aie-cols 2` to fall back to 8 cores if needed.
set -uo pipefail
source ~/src/mlir-aie/aie-env314.sh
EX=~/src/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
cd "$EX"
echo "=== xrt-smi device ==="
/opt/xilinx/xrt/bin/unwrapped/xrt-smi examine 2>/dev/null | grep -A2 "Device(s) Present" || true
echo "=== whole-array matmul on the NPU (JIT-compiles first time) ==="
timeout 900 python3 whole_array.py "$@"
echo "RUN_RC=$?"
