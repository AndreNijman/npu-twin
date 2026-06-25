"""Llama-3.2-1B-Instruct inference whose every weight matmul runs on the
XDNA1 Phoenix NPU (bf16 GEMV), orchestrated from Python.

The transformer's parameter-bearing compute — q/k/v/o projections, the SwiGLU
gate/up/down, and the (tied) lm_head — is dispatched to the AIE2 array via
``npu_gemv.gemv``. The lightweight, parameter-free glue (RMSNorm, llama3 RoPE,
GQA attention scores/softmax, SiLU, residual adds, argmax) runs on the CPU in
float32. >99% of the model's FLOPs and 100% of its 1.24B parameters are consumed
by matmuls that execute on the NPU.

Backends:
  - "npu" : weight matmuls on the Phoenix NPU (bf16 in, f32 out).
  - "cpu" : bit-faithful CPU emulation of the same bf16 GEMV (cast W,x->bf16,
            accumulate in f32) — the golden reference for NPU correctness.
  - "cpu_fp32" : full-precision reference (no bf16 rounding).

Usage:
  source ~/src/mlir-aie/aie-env314.sh
  python3 llama_npu.py --backend npu --prompt "..." --max-new 30
"""

import argparse
import sys
import time

import numpy as np
from ml_dtypes import bfloat16
from safetensors import safe_open
from tokenizers import Tokenizer

MODEL_DIR = None  # set in main


# ----------------------------------------------------------------------------
# config (Llama-3.2-1B-Instruct)
# ----------------------------------------------------------------------------
class Cfg:
    hidden = 2048
    n_layers = 16
    n_heads = 32
    n_kv_heads = 8
    head_dim = 64
    intermediate = 8192
    vocab = 128256
    eps = 1e-5
    rope_theta = 500000.0
    # llama3 rope scaling
    rope_factor = 32.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    old_context = 8192


# ----------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------
class Weights:
    def __init__(self, model_dir):
        self.f = safe_open(f"{model_dir}/model.safetensors", "numpy")
        self._cache = {}

    def get(self, name):
        if name not in self._cache:
            self._cache[name] = self.f.get_tensor(name)  # bf16 ndarray
        return self._cache[name]

    def layer(self, i, sub):
        return self.get(f"model.layers.{i}.{sub}.weight")


# ----------------------------------------------------------------------------
# llama3 rope inverse frequencies (matches HF _compute_llama3_parameters)
# ----------------------------------------------------------------------------
def llama3_inv_freq(cfg: Cfg):
    dim = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    factor = cfg.rope_factor
    low_freq_wavelen = cfg.old_context / cfg.low_freq_factor
    high_freq_wavelen = cfg.old_context / cfg.high_freq_factor
    wavelen = 2 * np.pi / inv_freq
    inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (cfg.old_context / wavelen - cfg.low_freq_factor) / (
        cfg.high_freq_factor - cfg.low_freq_factor
    )
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    inv_freq_llama = np.where(is_medium, smoothed, inv_freq_llama)
    return inv_freq_llama.astype(np.float64)  # [head_dim/2]


def rope_cos_sin(pos, inv_freq):
    # freqs[i] = pos * inv_freq[i]; emb = [freqs, freqs] -> [head_dim]
    freqs = pos * inv_freq  # [hd/2]
    emb = np.concatenate([freqs, freqs])  # [hd]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(vec, cos, sin):
    # vec: [..., head_dim]; rotate_half convention (HF)
    hd = vec.shape[-1]
    half = hd // 2
    rot = np.concatenate([-vec[..., half:], vec[..., :half]], axis=-1)
    return vec * cos + rot * sin


# ----------------------------------------------------------------------------
# matmul backends
# ----------------------------------------------------------------------------
def make_matmul(backend):
    if backend == "npu":
        import npu_gemv
        def mm(W_bf16, x_f32):
            return npu_gemv.gemv(W_bf16, x_f32.astype(bfloat16))
        return mm
    elif backend == "cpu":
        # bit-faithful emulation of the NPU's bf16-in/f32-acc GEMV
        def mm(W_bf16, x_f32):
            xb = x_f32.astype(bfloat16).astype(np.float32)
            return W_bf16.astype(np.float32) @ xb
        return mm
    elif backend == "cpu_fp32":
        def mm(W_bf16, x_f32):
            return W_bf16.astype(np.float32) @ x_f32
        return mm
    raise ValueError(backend)


# ----------------------------------------------------------------------------
# building blocks (CPU, fp32)
# ----------------------------------------------------------------------------
def rmsnorm(x, w_bf16, eps):
    x = x.astype(np.float32)
    var = np.mean(x * x)
    xn = x / np.sqrt(var + eps)
    return xn * w_bf16.astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class LlamaNPU:
    def __init__(self, model_dir, backend):
        self.cfg = Cfg()
        self.w = Weights(model_dir)
        self.mm = make_matmul(backend)
        self.backend = backend
        self.inv_freq = llama3_inv_freq(self.cfg)
        self.embed = self.w.get("model.embed_tokens.weight")  # [vocab, hidden] bf16
        # KV cache: per layer lists of [n_kv_heads, head_dim] arrays
        c = self.cfg
        self.k_cache = [[] for _ in range(c.n_layers)]
        self.v_cache = [[] for _ in range(c.n_layers)]
        self.pos = 0

    def reset(self):
        c = self.cfg
        self.k_cache = [[] for _ in range(c.n_layers)]
        self.v_cache = [[] for _ in range(c.n_layers)]
        self.pos = 0

    def forward_token(self, token_id):
        """One position. Returns logits[vocab] (float32). Mutates KV cache."""
        c = self.cfg
        pos = self.pos
        cos, sin = rope_cos_sin(pos, self.inv_freq)
        h = self.embed[token_id].astype(np.float32)  # [hidden]

        for li in range(c.n_layers):
            # --- attention ---
            hn = rmsnorm(h, self.w.layer(li, "input_layernorm"), c.eps)
            q = self.mm(self.w.layer(li, "self_attn.q_proj"), hn)   # [2048]
            k = self.mm(self.w.layer(li, "self_attn.k_proj"), hn)   # [512]
            v = self.mm(self.w.layer(li, "self_attn.v_proj"), hn)   # [512]

            q = q.reshape(c.n_heads, c.head_dim)
            k = k.reshape(c.n_kv_heads, c.head_dim)
            v = v.reshape(c.n_kv_heads, c.head_dim)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            self.k_cache[li].append(k)
            self.v_cache[li].append(v)
            K = np.stack(self.k_cache[li], axis=0)  # [T, n_kv, hd]
            V = np.stack(self.v_cache[li], axis=0)  # [T, n_kv, hd]

            scale = 1.0 / np.sqrt(c.head_dim)
            group = c.n_heads // c.n_kv_heads  # 4
            attn_out = np.empty((c.n_heads, c.head_dim), dtype=np.float32)
            for hi in range(c.n_heads):
                kvh = hi // group
                scores = (K[:, kvh, :] @ q[hi]) * scale  # [T]
                p = softmax(scores)
                attn_out[hi] = p @ V[:, kvh, :]  # [hd]
            attn_flat = attn_out.reshape(-1)  # [2048]
            o = self.mm(self.w.layer(li, "self_attn.o_proj"), attn_flat)
            h = h + o

            # --- mlp (swiglu) ---
            hn2 = rmsnorm(h, self.w.layer(li, "post_attention_layernorm"), c.eps)
            g = self.mm(self.w.layer(li, "mlp.gate_proj"), hn2)  # [8192]
            u = self.mm(self.w.layer(li, "mlp.up_proj"), hn2)    # [8192]
            act = (silu(g) * u).astype(np.float32)
            d = self.mm(self.w.layer(li, "mlp.down_proj"), act)  # [2048]
            h = h + d

        hn = rmsnorm(h, self.w.get("model.norm.weight"), c.eps)
        logits = self.mm(self.embed, hn)  # tied lm_head [vocab]
        self.pos += 1
        return logits

    def generate(self, token_ids, max_new, eos_ids, verbose=True):
        # prefill: feed all but the last prompt token, then loop
        logits = None
        for t in token_ids:
            logits = self.forward_token(int(t))
        out = []
        for _ in range(max_new):
            nxt = int(np.argmax(logits))
            out.append(nxt)
            if nxt in eos_ids:
                break
            logits = self.forward_token(nxt)
        return out


def main():
    global MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/andre/models/llama-3.2-1b-instruct")
    ap.add_argument("--backend", choices=["npu", "cpu", "cpu_fp32"], default="npu")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--chat", action="store_true", help="wrap prompt in Llama-3 chat template")
    ap.add_argument("--max-new", type=int, default=30)
    ap.add_argument("--raw", action="store_true", help="no BOS / no chat template")
    args = ap.parse_args()
    MODEL_DIR = args.model_dir

    tok = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    if args.chat:
        text = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{args.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        ids = tok.encode(text, add_special_tokens=False).ids
    elif args.raw:
        ids = tok.encode(args.prompt, add_special_tokens=False).ids
    else:
        ids = [128000] + tok.encode(args.prompt, add_special_tokens=False).ids  # BOS

    print(f"[backend={args.backend}] prompt_tokens={len(ids)}: {ids}", flush=True)
    model = LlamaNPU(args.model_dir, args.backend)
    eos_ids = {128001, 128008, 128009}

    t0 = time.time()
    out = model.generate(ids, args.max_new, eos_ids)
    dt = time.time() - t0
    n = len(out)
    txt = tok.decode(out)
    print(f"\n=== generated {n} tokens in {dt:.1f}s ({n/dt:.2f} tok/s) ===")
    print("TOKENS:", out)
    print("TEXT:", repr(txt))
    print("FULL:", repr(tok.decode(ids + out)))


if __name__ == "__main__":
    main()
