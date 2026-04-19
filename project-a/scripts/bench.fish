#!/usr/bin/env fish
# Benchmark runner: 20 prompts × {baseline via llama-completion, speculative via llama-speculative}.
# Output: bench/results/<UTC>.json   (committed; raw logs under raw/ — gitignored)
#
# Sampling pinned greedy (--temp 0, --seed 42) for reproducibility and best-case accept rate.

set -l here (status dirname)
set -g repo_root (realpath $here/../..)
set -g models "$repo_root/project-a/models"
set -g prompts "$repo_root/bench/prompts/suite.jsonl"

set -q A_TARGET_MODEL_FILE; or set -g A_TARGET_MODEL_FILE "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
set -q A_DRAFT_MODEL_FILE;  or set -g A_DRAFT_MODEL_FILE  "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
set -q A_CTX;               or set -g A_CTX 4096
set -q A_N_PREDICT;         or set -g A_N_PREDICT 160

set -g target "$models/$A_TARGET_MODEL_FILE"
set -g draft  "$models/$A_DRAFT_MODEL_FILE"

for f in $target $draft $prompts
    if not test -e $f
        echo "error: missing $f" >&2
        exit 1
    end
end
if not command -q llama-completion
    echo "error: llama-completion missing" >&2
    exit 1
end
if not command -q llama-speculative
    echo "error: llama-speculative missing" >&2
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

# ---------- runners ----------
function run_baseline --argument-names prompt log
    env AMD_VULKAN_ICD=RADV llama-completion \
        -m $target --device Vulkan0 -ngl 99 -fa on \
        -c $A_CTX --seed 42 --temp 0 -n $A_N_PREDICT \
        -p "$prompt" </dev/null >$log 2>&1
end

function run_speculative --argument-names prompt log
    env AMD_VULKAN_ICD=RADV llama-speculative \
        -m $target -md $draft \
        --device Vulkan0 -ngl 99 -devd Vulkan0 -ngld 99 \
        -fa on -c $A_CTX --seed 42 --temp 0 -n $A_N_PREDICT \
        --draft-max 8 --draft-min 2 --draft-p-min 0.6 \
        -p "$prompt" </dev/null >$log 2>&1
end

# ---------- parsers ----------
# Baseline (llama-completion): "<N> tokens per second" is the eval/gen rate.
# There are multiple "tokens per second" lines (prompt eval, eval). Take last.
function parse_tg_baseline --argument log
    grep -oE '[0-9]+\.[0-9]+ tokens per second' $log | tail -1 \
        | grep -oE '[0-9]+\.[0-9]+'
end

# Speculative (llama-speculative): "decoded N tokens in T seconds, speed: X t/s"
function parse_tg_speculative --argument log
    grep -oE 'decoded[[:space:]]+[0-9]+[[:space:]]+tokens in[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+seconds,[[:space:]]+speed:[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+t/s' $log \
        | tail -1 | grep -oE '[0-9]+\.[0-9]+[[:space:]]+t/s' | grep -oE '[0-9]+\.[0-9]+'
end

# Speculative: "accept    = 57.292%"
function parse_accept --argument log
    grep -oE '^accept[[:space:]]+=[[:space:]]+[0-9]+\.[0-9]+%' $log \
        | tail -1 | grep -oE '[0-9]+\.[0-9]+'
end

# ---------- run ----------
echo "bench utc=$utc  ctx=$A_CTX  n_predict=$A_N_PREDICT"
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
    run_baseline $prompt $base_log
    set -l tg_b (parse_tg_baseline $base_log)
    test -z "$tg_b"; and set tg_b "null"

    echo "[$idx/20] id=$pid class=$plabel speculative..."
    run_speculative $prompt $spec_log
    set -l tg_s (parse_tg_speculative $spec_log)
    test -z "$tg_s"; and set tg_s "null"
    set -l acc (parse_accept $spec_log)
    test -z "$acc"; and set acc "null"

    set -l speedup "null"
    if test $tg_b != "null" -a $tg_s != "null"
        set speedup (math -s3 $tg_s / $tg_b)
    end

    echo "    baseline=$tg_b t/s  spec=$tg_s t/s  accept=$acc%  speedup=$speedup×"

    set -l entry (jq -nc \
        --argjson id $pid \
        --arg class $plabel \
        --argjson tg_baseline $tg_b \
        --argjson tg_speculative $tg_s \
        --argjson accept_rate $acc \
        --argjson speedup $speedup \
        '{id:$id, class:$class, tg_baseline:$tg_baseline, tg_speculative:$tg_speculative, accept_rate:$accept_rate, speedup:$speedup}')
    set results (echo $results | jq -c ". + [$entry]")
end < $prompts

set -l hostn (hostnamectl hostname 2>/dev/null; or echo unknown)
set -l meta (jq -nc \
    --arg utc "$utc" \
    --arg target "$A_TARGET_MODEL_FILE" \
    --arg draft "$A_DRAFT_MODEL_FILE" \
    --argjson ctx $A_CTX \
    --argjson n_predict $A_N_PREDICT \
    --arg sampling "greedy (temp=0, seed=42)" \
    --arg device "Vulkan0 (RADV PHOENIX / 780M)" \
    --arg host "$hostn" \
    '{utc:$utc, target:$target, draft:$draft, ctx:$ctx, n_predict:$n_predict, sampling:$sampling, device:$device, host:$host}')

echo $results | jq --argjson meta "$meta" '{meta:$meta, results:.}' > $out_json
echo
echo "wrote $out_json"

echo
echo "class averages:"
jq -r '
    .results
    | group_by(.class)
    | map({
        class: .[0].class,
        n: length,
        avg_baseline: ([.[] | .tg_baseline | numbers] | if length>0 then add/length else null end),
        avg_spec:     ([.[] | .tg_speculative | numbers] | if length>0 then add/length else null end),
        avg_accept:   ([.[] | .accept_rate | numbers] | if length>0 then add/length else null end),
        avg_speedup:  ([.[] | .speedup | numbers] | if length>0 then add/length else null end)
      })
    | .[] | "  \(.class) n=\(.n)  base=\(.avg_baseline)  spec=\(.avg_spec)  accept=\(.avg_accept)%  speedup=\(.avg_speedup)×"
' $out_json

echo
set -l overall (jq -r '.results | map(.speedup | numbers) | add/length' $out_json)
echo "overall avg speedup: $overall ×"
