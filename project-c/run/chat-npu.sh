#!/usr/bin/env bash
# Interactive chat with Llama-3.2-1B-Instruct running on the XDNA1 Phoenix NPU.
# Every weight matmul executes on the NPU (bf16 GEMV); the CPU does only norm/
# rope/softmax/argmax. Multi-turn, streaming, KV-cache carried across turns.
#
#   bash project-c/run/chat-npu.sh                 # if already in the 'render' group
#   sg render -c 'bash project-c/run/chat-npu.sh'  # otherwise (grants /dev/accel access)
#
# Options pass straight through to llama_npu.py, e.g.:
#   bash project-c/run/chat-npu.sh --greedy
#   bash project-c/run/chat-npu.sh --temp 0.8 --top-p 0.95
#   bash project-c/run/chat-npu.sh --system "You are a terse pirate." --max-new 128
#
# In-chat commands: /reset (clear context)   /exit (quit)   Ctrl-D (quit)
#
# NB: ~0.2 tok/s (1B model, single AIE core) — replies stream in slowly. The 780M
# iGPU decodes this same model ~2.2x faster; the NPU's win is perf/watt + prefill.
set -uo pipefail
source ~/src/mlir-aie/aie-env314.sh
cd ~/src/npu-twin/project-c/m6-llm

# warn (don't block) if the device node isn't reachable
if [ ! -r /dev/accel/accel0 ]; then
  echo "WARNING: /dev/accel/accel0 not readable — re-run under: sg render -c 'bash project-c/run/chat-npu.sh'" >&2
fi

exec python3 -u llama_npu.py --interactive "$@"
