# Source this to run IRON/mlir-aie designs on the XDNA1 Phoenix NPU.
# Combines: Python 3.14 IRON venv (mlir_aie + Peano) + system XRT built from
# amd/xdna-driver (pyxrt + libxrt_coreutil + libxrt_driver_xdna shim).
# pyxrt is the runtime gate the IRON DefaultNPURuntime imports; the wheel does
# NOT bundle it, so it comes from /opt/xilinx/xrt (built against system py3.14).
source ~/src/mlir-aie/ironenv314/bin/activate
source /opt/xilinx/xrt/setup.sh                      # XILINX_XRT, pyxrt on PYTHONPATH, xrt libs
export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie 2>/dev/null | awk '/^Location:/{print $2}')/mlir_aie"
export PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/{print $2}')/llvm-aie"
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PATH"
# llvm-objcopy that understands the AIE2 ELF (Peano ships no objcopy; GNU objcopy
# rejects the AIE2 machine type). Appended so it only supplies llvm-objcopy.
export PATH="$PATH:/usr/lib/llvm/21/bin"
export PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib:$LD_LIBRARY_PATH"
# Put system XRT libs FIRST so pyxrt binds the matching libxrt_coreutil (2.25.0),
# not the older copy bundled inside the mlir_aie wheel.
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib64:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH"
export NPU2=0   # Phoenix = npu1 / AIE2
