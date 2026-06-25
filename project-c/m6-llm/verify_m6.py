"""M6 verification: prefill a prompt through Llama-3.2-1B twice — once with all
weight matmuls on the XDNA1 NPU (bf16 GEMV), once in full fp32 on the CPU — and
compare the final next-token logits. The NPU run has no CPU-fallback path
(npu_gemv opens /dev/accel/accel0 and raises if absent), so agreement here is
proof the math ran on the AIE2 array and is numerically faithful.
"""
import sys
import time
import numpy as np
from llama_npu import LlamaNPU, Cfg
from tokenizers import Tokenizer

MODEL_DIR = "/home/andre/models/llama-3.2-1b-instruct"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"


def prefill_logits(backend, ids):
    m = LlamaNPU(MODEL_DIR, backend)
    t0 = time.time()
    logits = None
    for t in ids:
        logits = m.forward_token(int(t))
    return logits, time.time() - t0


def main():
    tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    ids = [128000] + tok.encode(PROMPT, add_special_tokens=False).ids
    print(f"prompt: {PROMPT!r}  tokens={ids}")

    lc, tc = prefill_logits("cpu_fp32", ids)
    print(f"[cpu_fp32] prefill {len(ids)} tok in {tc:.1f}s")
    ln, tn = prefill_logits("npu", ids)
    print(f"[npu]      prefill {len(ids)} tok in {tn:.1f}s")

    # metrics
    cos = float(ln @ lc / (np.linalg.norm(ln) * np.linalg.norm(lc)))
    rel = float(np.linalg.norm(ln - lc) / np.linalg.norm(lc))
    top1_c = int(np.argmax(lc)); top1_n = int(np.argmax(ln))
    top5_c = np.argsort(lc)[-5:][::-1]
    top5_n = np.argsort(ln)[-5:][::-1]
    print(f"\nlogits cosine(npu,cpu_fp32) = {cos:.6f}")
    print(f"logits relL2                = {rel:.4e}")
    print(f"argmax  cpu_fp32 = {top1_c} ({tok.decode([top1_c])!r})")
    print(f"argmax  npu      = {top1_n} ({tok.decode([top1_n])!r})")
    print(f"top5 cpu_fp32 = {top5_c.tolist()}")
    print(f"top5 npu      = {top5_n.tolist()}")
    ok = (top1_c == top1_n) and cos > 0.999 and (top5_c.tolist() == top5_n.tolist())
    print("\nVERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
