#!/usr/bin/env fish
# Fetch GGUF weights for Project A.
# Models land in project-a/models/ (gitignored). Safe to re-run — hf caches.

set -l here (status dirname)
set -l repo_root (realpath $here/../..)
set -l dest "$repo_root/project-a/models"

set -q A_TARGET_MODEL_REPO; or set -l A_TARGET_MODEL_REPO "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
set -q A_TARGET_MODEL_FILE; or set -l A_TARGET_MODEL_FILE "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
set -q A_DRAFT_MODEL_REPO;  or set -l A_DRAFT_MODEL_REPO  "bartowski/Llama-3.2-1B-Instruct-GGUF"
set -q A_DRAFT_MODEL_FILE;  or set -l A_DRAFT_MODEL_FILE  "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

if not command -q hf
    echo "error: hf CLI missing. Install: sudo pacman -S python-huggingface-hub" >&2
    exit 1
end

mkdir -p $dest

echo "==> target: $A_TARGET_MODEL_REPO :: $A_TARGET_MODEL_FILE"
hf download $A_TARGET_MODEL_REPO $A_TARGET_MODEL_FILE --local-dir $dest
or begin; echo "target fetch failed" >&2; exit 1; end

echo "==> draft:  $A_DRAFT_MODEL_REPO :: $A_DRAFT_MODEL_FILE"
hf download $A_DRAFT_MODEL_REPO $A_DRAFT_MODEL_FILE --local-dir $dest
or begin; echo "draft fetch failed" >&2; exit 1; end

echo
echo "Downloaded:"
ls -lh $dest/$A_TARGET_MODEL_FILE $dest/$A_DRAFT_MODEL_FILE
