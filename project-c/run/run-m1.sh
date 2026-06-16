#!/usr/bin/env bash
# M1 proof: run the IRON passthrough_kernel on the XDNA1 Phoenix NPU.
set -uo pipefail
source ~/src/mlir-aie/aie-env314.sh
echo "=== sanity: pyxrt + aie import ==="
python -c "import pyxrt; print('pyxrt OK:', pyxrt.__file__)" || exit 31
python -c "import aie; print('aie OK:', aie.__file__)" || exit 32
echo "=== xrt-smi sees device ==="
/opt/xilinx/xrt/bin/unwrapped/xrt-smi examine 2>/dev/null | grep -A2 "Device(s) Present" || true
echo "=== RUN passthrough_kernel on the NPU (JIT-compiles first time) ==="
cd ~/src/mlir-aie/programming_examples/basic/passthrough_kernel
timeout 480 python3 passthrough_kernel.py -i1s 4096
echo "RUN_RC=$?"
