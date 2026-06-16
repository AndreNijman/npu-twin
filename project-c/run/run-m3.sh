#!/usr/bin/env bash
# M3 proof: 1x1 int8 conv2d on the XDNA1 Phoenix NPU, verified against a PyTorch
# golden model (programming_examples/ml/conv2d, kernels.conv2dk1_i8, Peano).
# Defaults: 32x32, 64 in-channels -> 64 out-channels, signed int8 in/out.
# Needs `torch` (CPU) in the venv: pip install torch --index-url https://download.pytorch.org/whl/cpu
set -uo pipefail
source ~/src/mlir-aie/aie-env314.sh
EX=~/src/mlir-aie/programming_examples/ml/conv2d
cd "$EX"
echo "=== build int8 conv2d xclbin (Peano) + run on NPU + torch-golden verify ==="
timeout 600 make run_py "$@"
echo "RUN_RC=$?"
