# ADR-0003: Llama 3.1 8B Instruct (target) + Llama 3.2 1B Instruct (draft)

Date: 2026-04-19
Status: Accepted

## Context

Speculative decoding only pays off when the draft model's outputs are
accepted often enough by the target. Low acceptance wastes draft compute
and KV; high acceptance gives a meaningful speedup.

## Decision

Use **`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (Q4_K_M)** as target and
**`bartowski/Llama-3.2-1B-Instruct-GGUF` (Q4_K_M)** as draft.

## Reasoning

- Same tokenizer (Llama 3 family / `tiktoken`-style BPE). `llama.cpp`'s
  draft path requires tokenizer compatibility.
- Same base vocab (Llama 3.1 and 3.2 share the vocab). No mismatch error.
- Instruct-tuned on similar data. The 1B has a high accept rate on common
  prose and boilerplate code patterns, which is where speculation shines.
- Bartowski's quants are well-calibrated and publicly distributed without
  gating. Reproducible.
- Q4_K_M on both: good balance of memory (5.4 GB combined weights) and
  quality; avoids Q4_0 artifacts.

## Alternatives / fallback chain

1. **Plan B2 (Qwen2.5-Coder pair):** If accept rate on code prompts is
   low with Llama 3.1/3.2, switch to
   `bartowski/Qwen2.5-Coder-7B-Instruct-GGUF` +
   `bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF` (same tokenizer, tighter
   code distribution). Log the switch as ADR-0004.
2. **Plan B3 (n-gram self-speculative):** If a usable draft model cannot
   be matched to the target, use `--spec-type ngram-mod` — no separate
   draft weights needed.

## Consequences

- Combined RAM footprint stable; both GGUFs live in `project-a/models/`
  (gitignored).
- Bench suite (20 prompts, 4 classes) produces one `bench/results/*.json`
  run that spans both configurations; acceptance criterion is
  `mean(tg_speculative) / mean(tg_baseline) ≥ 1.3` on the `code` class.
- If the acceptance threshold is missed, the fallback chain above is
  deterministic and each switch is logged in a new ADR.
