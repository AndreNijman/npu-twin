# M6 — a 1B-parameter LLM on the XDNA1 NPU (open stack, no Vitis)

**Phase 8, milestone M6 (2026-06-25):** **Llama-3.2-1B-Instruct** generates text
with **every weight matmul executed on the ThinkPad L16's AMD XDNA1 "Phoenix"
NPU**, through the open IRON / mlir-aie / Peano stack — no AMD Vitis, no VitisAI
EP. This is the assembled-LLM frontier the project's research flagged: mlir-aie
ships the transformer building blocks but no LLM; this wires them into one.

## The result

```
$ python3 llama_npu.py --backend npu --raw --prompt "The capital of France is" --max-new 6
[backend=npu] prompt_tokens=5: [791, 6864, 315, 9822, 374]
=== generated 6 tokens in 35.9s (0.17 tok/s) ===
TEXT: ' Paris. The capital of Germany'
```

A chat-templated instruction, generated the same way (every matmul on the NPU):

```
$ python3 llama_npu.py --backend npu --chat --prompt "In one sentence, what is an NPU?" --max-new 40
"An NPU (Numerical Processing Unit) is a measure of the processing power of a
 computer's central processing unit (CPU), indicating how many calculations it
 can perform per second."          # fluent + ends on EOS (the definition is the
                                   # 1B model's own knowledge limit, not a bug)
```

## Chat with it (interactive)

```bash
bash project-c/run/chat-npu.sh                 # or: sg render -c 'bash project-c/run/chat-npu.sh'
```

A multi-turn REPL — every matmul on the NPU, replies **stream token-by-token**,
the KV cache is carried across turns. In-chat commands: `/reset` (clear context),
`/exit` (quit), Ctrl-D. Flags pass through (`--greedy`, `--temp 0.8 --top-p 0.95`,
`--system "You are a terse pirate."`, `--max-new 128`).

```
Llama-3.2-1B-Instruct on the XDNA1 NPU  (every matmul on the Phoenix NPU, bf16; greedy)
you> Hi! Reply in 5 words.
bot> I'm here to help.
[6 tok, 17s, 0.36 tok/s]
you> Name one primary color.
bot> Red.
[2 tok, 7s, 0.28 tok/s]
```

Expect **~0.2 tok/s** (1B model, single AIE core) — replies stream in slowly,
that's the honest speed (see scope below). The streaming decode holds back
incomplete multi-byte UTF-8 (emoji/CJK) until the bytes complete, so output is
never garbled (hardened after an adversarial review caught the naive delta-print
corrupting split emoji).

## What runs where

The transformer's **parameter-bearing compute runs on the NPU**; the
parameter-free glue runs on the CPU in f32.

| On the **NPU** (bf16 GEMV, AIE2) | On the **CPU** (f32) |
|---|---|
| q / k / v / o projections | RMSNorm |
| SwiGLU gate / up / down | llama3 RoPE |
| tied **lm_head** (128256×2048) | GQA attention scores + softmax |
| — | SiLU, residual adds, argmax |

That is **100% of the model's 1.24B parameters** and **>99% of its FLOPs**. Each
`nn.Linear` is a matrix-vector product `y[M] = W[M,K] @ x[K]`; the HF weight is
stored `[out, in] = [M, K]`, which is exactly the `A` matrix the
`matvec_vectorized_bf16_f32` kernel consumes — no transpose.

## Why bf16 (and how it was enabled)

An `int16 × int16` GEMV summed over K=8192 (down_proj / the residual width)
**overflows the int32 accumulator** (8192 × ~1e9 ≫ 2³¹) — that would cap useful
precision at ~8 bits, too lossy for a coherent 1B model. **bf16 inputs with an
f32 accumulator** are the model's native dtype and overflow-free; Phoenix AIE2
does bf16 MAC in hardware (M5's Magika used bf16).

mlir-aie's `mv` kernel supports bf16 in C++ but ships it **commented out** with
the Python wrapper locked to int16. Two tiny edits enable it — see
[`PATCH-bf16-gemv.md`](PATCH-bf16-gemv.md) (candidate upstream PR).

## Verification

`verify_m6.py` prefills a prompt through the model twice — all matmuls on the
NPU vs full fp32 on the CPU — and compares the next-token logits. There is **no
CPU-fallback path**: `npu_gemv` opens `/dev/accel/accel0` and raises if absent.

```
logits cosine(npu, cpu_fp32) = 0.999992
logits relL2                 = 3.9e-3
argmax: 12366 (' Paris')  ==  12366 (' Paris')
top5:   [12366, 264, 539, 1131, 1101]  ==  [12366, 264, 539, 1131, 1101]
VERIFY: PASS
```

Per-shape bf16 GEMV vs numpy (every Llama Linear, incl. the 128k lm_head):
relL2 ~1e-7…1e-6, cos=1.000000 (`../proof/m6-gemv-bf16-shapes.txt`).

Proof artifacts in [`../proof/`](../proof/): `m6-llm-run.txt` (generation +
verify), `m6-chat-demo.txt`, `m6-gemv-bf16-shapes.txt`, `m6-gemv-bf16-aie2.disasm`
(the bf16 core ELF is `EM_AIE` AIE2 VLIW), `m6-xrt-smi.txt`.

## Honest scope — this is "it runs and is correct", not a speed result

~0.17 tok/s. It is **single-core**, **re-uploads each weight host→device every
call**, and issues **~255 kernel invocations per token** (M-blocking × 16 layers
× the 7 Linears + the 63-block lm_head). Decode is DRAM-bandwidth-bound and the
**780M iGPU runs this same model ~2.2× faster** (≈44 tok/s, llama.cpp Vulkan).
The NPU's real LLM wins are **prefill/TTFT** and **perf/watt** — not decode
throughput. Speedups left on the table: resident on-device weights, whole-array
(16-tile) GEMV, on-NPU norm/rope/softmax. See
[ADR-0010](../../docs/decisions/0010-1b-llm-on-xdna1-npu.md).

**M-blocking.** The B-vector DMA repeats the activation `M/m` times through one
buffer-descriptor wrap dimension capped at 64 ⇒ one kernel call is limited to
`M ≤ 2048` rows; larger Linears are split on the host and concatenated. K is
unconstrained (down_proj K=8192 compiles directly).

## Files

| File | What |
|------|------|
| `npu_gemv.py` | bf16 GEMV on the NPU (IRON design + M-blocked host wrapper); `python3 npu_gemv.py [--full]` self-tests vs numpy |
| `llama_npu.py` | Llama-3.2-1B forward + KV-cache generation; `--backend {npu,cpu,cpu_fp32}`; `--interactive` multi-turn streaming chat REPL |
| `../run/chat-npu.sh` | launcher for the interactive chat (`--interactive`) on the NPU |
| `verify_m6.py` | NPU-vs-fp32 logits gate |
| `debug_gemv.py` | parametric GEMV harness (dtype × scalar/vectorized) used to bring up bf16 |
| `PATCH-bf16-gemv.md` | the two mv.cc / linalg.py edits that enable bf16 GEMV |

## Reproduce

```bash
# one-time: enable bf16 GEMV (PATCH-bf16-gemv.md), then:
pip install tokenizers safetensors            # into the IRON venv
# fetch unsloth/Llama-3.2-1B-Instruct (bf16) -> ~/models/llama-3.2-1b-instruct
source ~/src/mlir-aie/aie-env314.sh
sg render -c 'bash ../run/run-m6.sh'           # one-shot demo
sg render -c 'bash ../run/run-m6.sh --verify'  # NPU vs fp32 logits
sg render -c 'bash ../run/chat-npu.sh'         # interactive multi-turn chat
```

## Hardware / versions (this run)

- ThinkPad L16 Gen 2 — Ryzen 7 PRO 250, **XDNA1 "Phoenix"** NPU (`1022:1502`, aie2, 6×5)
- Void Linux, kernel **7.1.0**, glibc 2.41; in-tree `amdxdna` + firmware `npu.sbin.1.5.5.391`
- XRT **2.25.0** (from `amd/xdna-driver`, `-nokmod`), `mlir_aie` 1.3.3.dev8, Peano (llvm-aie) 21.0, Python 3.14
- Model: `unsloth/Llama-3.2-1B-Instruct` (bf16 safetensors, tied embeddings, llama3 RoPE)
