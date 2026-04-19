#!/usr/bin/env fish
# Benchmark runner: 20 prompts × {with draft, without draft}.
# Output: bench/results/<UTC>.json   (committed; raw logs under raw/ — gitignored)

set -l here (status dirname)
set -l repo_root (realpath $here/../..)
set -l models "$repo_root/project-a/models"
set -l prompts "$repo_root/bench/prompts/suite.jsonl"

set -q A_TARGET_MODEL_FILE; or set -l A_TARGET_MODEL_FILE "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
set -q A_DRAFT_MODEL_FILE;  or set -l A_DRAFT_MODEL_FILE  "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
set -q A_CTX;               or set -l A_CTX 4096

set -l target "$models/$A_TARGET_MODEL_FILE"
set -l draft  "$models/$A_DRAFT_MODEL_FILE"

for f in $target $draft $prompts
    if not test -e $f
        echo "error: missing $f" >&2
        exit 1
    end
end
if not command -q llama-cli
    echo "error: llama-cli missing" >&2
    exit 1
end
if not command -q jq
    echo "error: jq missing. sudo pacman -S jq" >&2
    exit 1
end

set -l utc (date -u +%Y%m%dT%H%M%SZ)
set -l raw_dir "$repo_root/bench/results/raw/$utc"
set -l out_json "$repo_root/bench/results/$utc.json"
mkdir -p $raw_dir

# ---------- helpers ----------
function run_once --argument-names mode pid plabel prompt log
    # mode: baseline | speculative
    set -l cmd /usr/bin/llama-cli -m $target --device Vulkan0 -ngl 99 -fa on \
        -c $A_CTX --seed 42 -t 12 -n 160 -no-cnv -p "$prompt"
    if test $mode = speculative
        set -a cmd -md $draft -devd Vulkan0 -ngld 99 \
            --draft-max 8 --draft-min 2 --draft-p-min 0.6
    end
    env AMD_VULKAN_ICD=RADV $cmd 2>&1 > $log
end

function parse_tg --argument log
    # llama.cpp prints: "eval time ... N tokens (   T ms per token,   R tokens per second)"
    grep -oE 'eval time.*tokens per second' $log | tail -1 \
        | grep -oE '[0-9]+\.[0-9]+ tokens per second' | head -1 \
        | grep -oE '[0-9]+\.[0-9]+'
end

function parse_accept --argument log
    # speculative prints: "draft acceptance rate: X.XX"
    set -l rate (grep -oE 'acceptance rate[^0-9]*[0-9.]+' $log | tail -1 | grep -oE '[0-9.]+$')
    if test -z "$rate"
        set rate "null"
    end
    echo $rate
end

# ---------- run ----------
echo "bench utc=$utc"
set -l results '[]'
set -l idx 0

while read -l line
    set idx (math $idx + 1)
    set -l pid    (echo $line | jq -r .id)
    set -l plabel (echo $line | jq -r .class)
    set -l prompt (echo $line | jq -r .prompt)

    set -l base_log "$raw_dir/p$pid-baseline.log"
    set -l spec_log "$raw_dir/p$pid-speculative.log"

    echo "[$idx/20] id=$pid class=$plabel baseline..."
    run_once baseline $pid $plabel $prompt $base_log
    set -l tg_b (parse_tg $base_log)
    test -z "$tg_b"; and set tg_b "null"

    echo "[$idx/20] id=$pid class=$plabel speculative..."
    run_once speculative $pid $plabel $prompt $spec_log
    set -l tg_s (parse_tg $spec_log)
    test -z "$tg_s"; and set tg_s "null"
    set -l acc (parse_accept $spec_log)

    set -l speedup "null"
    if test $tg_b != "null" -a $tg_s != "null"
        set speedup (math -s3 $tg_s / $tg_b)
    end

    set -l entry (jq -n \
        --argjson id $pid \
        --arg class $plabel \
        --argjson tg_baseline $tg_b \
        --argjson tg_speculative $tg_s \
        --arg accept_rate "$acc" \
        --arg speedup "$speedup" \
        '{id:$id, class:$class, tg_baseline:$tg_baseline, tg_speculative:$tg_speculative, accept_rate:$accept_rate, speedup:$speedup}')
    set results (echo $results | jq ". + [$entry]")
end < $prompts

set -l meta (jq -n \
    --arg utc "$utc" \
    --arg target "$A_TARGET_MODEL_FILE" \
    --arg draft "$A_DRAFT_MODEL_FILE" \
    --argjson ctx $A_CTX \
    --arg device "Vulkan0 (RADV PHOENIX / 780M)" \
    --arg host (hostnamectl hostname 2>/dev/null | string collect) \
    '{utc:$utc, target:$target, draft:$draft, ctx:$ctx, device:$device, host:$host}')

echo $results | jq --argjson meta $meta '{meta:$meta, results:.}' > $out_json
echo
echo "wrote $out_json"
jq '.results | map(.speedup|tonumber? // null) | map(select(.!=null)) | add / length' $out_json | string trim | read -l avg
echo "average speedup: $avg ×"
