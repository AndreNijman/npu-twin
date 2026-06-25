#!/usr/bin/env bash
# M6 proof: Llama-3.2-1B-Instruct generates text with EVERY weight matmul
# (q/k/v/o, gate/up/down, tied lm_head) executed on the XDNA1 Phoenix NPU as a
# bf16 GEMV (matvec_vectorized_bf16_f32) on one AIE2 core. The CPU does only the
# parameter-free glue: RMSNorm, llama3 RoPE, GQA attention scores/softmax, SiLU,
# residual adds, argmax. No CPU-fallback path — npu_gemv opens /dev/accel/accel0
# and raises if the device is absent.
#
#   sg render -c 'bash project-c/run/run-m6.sh'            # default demo
#   sg render -c 'bash project-c/run/run-m6.sh --verify'   # NPU vs fp32 logits
#
# One-time: enable the bf16 GEMV kernel (project-c/m6-llm/PATCH-bf16-gemv.md) and
# fetch the model + python deps:
#   pip install tokenizers safetensors          # into the IRON venv
#   huggingface_hub: unsloth/Llama-3.2-1B-Instruct -> ~/models/llama-3.2-1b-instruct
set -uo pipefail
source ~/src/mlir-aie/aie-env314.sh
EX=~/src/npu-twin/project-c/m6-llm
cd "$EX"
echo "=== xrt-smi device ==="
/opt/xilinx/xrt/bin/unwrapped/xrt-smi examine 2>/dev/null | grep -A2 "Device(s) Present" || true

if [ "${1:-}" = "--verify" ]; then
  echo "=== M6 verify: NPU forward vs full fp32 forward (final-token logits) ==="
  exec python3 -u verify_m6.py "${2:-The capital of France is}"
fi

echo "=== Llama-3.2-1B-Instruct on the NPU (greedy) ==="
PROMPT="${PROMPT:-The capital of France is}"
exec python3 -u llama_npu.py --backend npu --raw --prompt "$PROMPT" --max-new "${MAXNEW:-12}"
