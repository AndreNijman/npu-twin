#!/usr/bin/env fish
# Standalone llama-completion smoke test on 780M Vulkan.
# Confirms the target model loads and generates without the server.

set -l here (status dirname)
set -l repo_root (realpath $here/../..)
set -l models "$repo_root/project-a/models"

set -q A_TARGET_MODEL_FILE; or set -l A_TARGET_MODEL_FILE "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

set -l target "$models/$A_TARGET_MODEL_FILE"

if not test -f $target
    echo "error: target model missing: $target" >&2
    echo "Run: project-a/scripts/fetch-models.fish" >&2
    exit 1
end

if not command -q llama-completion
    echo "error: llama-completion missing. Install: paru -S llama.cpp-vulkan" >&2
    exit 1
end

set -l log /tmp/npu-twin-smoke.log
echo "==> smoke test (target only, 780M Vulkan). Log: $log"
env AMD_VULKAN_ICD=RADV \
    llama-completion \
        -m $target \
        --device Vulkan0 \
        -ngl 99 \
        -fa on \
        -c 1024 \
        -n 48 \
        --seed 42 \
        -p "Write a short haiku about a quiet lake." >$log 2>&1

if grep -qE 'graph splits[[:space:]]*=[[:space:]]*1\b' $log
    echo "ok: graph splits = 1 (all tensors on GPU)"
else if grep -q 'graph splits' $log
    echo "warn: graph splits != 1 — some ops falling back to CPU"
    grep 'graph splits' $log
else
    echo "warn: 'graph splits' line not found — llama.cpp output format may differ"
end

grep -oE 'tokens per second' $log >/dev/null; and grep -oE '[0-9.]+[[:space:]]+tokens per second' $log | head -4
grep -E 'Generation:' $log | head -2
