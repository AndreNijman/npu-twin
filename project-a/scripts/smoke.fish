#!/usr/bin/env fish
# Standalone llama-cli smoke test on 780M Vulkan.
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

if not command -q llama-cli
    echo "error: llama-cli missing. Install: paru -S llama.cpp-vulkan" >&2
    exit 1
end

echo "==> smoke test (target only, 780M Vulkan)"
env AMD_VULKAN_ICD=RADV \
    llama-cli \
        -m $target \
        --device Vulkan0 \
        -ngl 99 \
        -fa on \
        -c 1024 \
        -no-cnv \
        -n 32 \
        -p "Say hi in five words." 2>&1 | tee /tmp/npu-twin-smoke.log

set -l ok 1
if not grep -q 'graph splits' /tmp/npu-twin-smoke.log
    echo "warn: 'graph splits' not in output — llama.cpp API may have changed" >&2
    set ok 0
end

if grep -qE 'graph splits *= *1\b' /tmp/npu-twin-smoke.log
    echo "ok: graph splits = 1 (all tensors on GPU)"
else
    echo "warn: graph splits != 1 — some ops falling back to CPU" >&2
end
