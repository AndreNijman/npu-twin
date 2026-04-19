#!/usr/bin/env fish
# npu-twin end-to-end demo.
#
# One short pass that exercises both projects on a freshly-logged-in box:
#   1. preflight (kernel, NPU, accel, RADV, video group, /dev/video0)
#   2. both --user services active (llama-speculative + presenced)
#   3. one short llama-speculative prompt — prints tok/s + accept-rate
#   4. presenced status — service state + last FSM transition from journal
#
# Non-destructive. Reads state, does not mutate config. Exit 0 on all-pass,
# non-zero on first hard fail. Meant to be the "does it work?" smoke gate
# referenced from README and the Phase 5 acceptance criteria.

set -l here (status dirname)
set -g repo_root (realpath $here/..)
set -g models "$repo_root/project-a/models"
set -g fail 0

function ok   ; echo "  [ok]   $argv" ; end
function warn ; echo "  [warn] $argv" ; end
function bad  ; echo "  [FAIL] $argv" ; set -g fail 1 ; end

echo "== npu-twin demo =="
echo "repo: $repo_root"
echo

# --- 1. preflight ---
# preflight.fish: exit 0 = pass, 1 = hard fail, 2 = warn-only (relog
# pending). Treat warn as non-fatal for the demo gate.
echo "[1/4] preflight"
fish "$repo_root/scripts/preflight.fish" > /tmp/npu-twin-demo-preflight.log 2>&1
set -l pf_rc $status
switch $pf_rc
    case 0
        ok "preflight clean (see /tmp/npu-twin-demo-preflight.log)"
    case 2
        warn "preflight warn-only (rc=2, see /tmp/npu-twin-demo-preflight.log)"
    case '*'
        bad "preflight failed rc=$pf_rc — see /tmp/npu-twin-demo-preflight.log"
end
echo

# --- 2. services ---
echo "[2/4] --user services"
for svc in llama-speculative.service presenced.service
    set -l state (systemctl --user is-active $svc 2>/dev/null)
    if test "$state" = "active"
        ok "$svc active"
    else
        bad "$svc not active (state=$state). install with systemctl --user link + daemon-reload + enable --now"
    end
end
echo

# --- 3. short llama-speculative prompt ---
echo "[3/4] llama-speculative one-shot"
set -l target "$models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
set -l draft  "$models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
if not test -e $target; or not test -e $draft
    bad "models missing ($target or $draft)"
else
    set -l log /tmp/npu-twin-demo-spec.log
    env AMD_VULKAN_ICD=RADV llama-speculative \
        -m $target -md $draft \
        --device Vulkan0 -ngl 99 -devd Vulkan0 -ngld 99 \
        -fa on -c 2048 --seed 42 --temp 0 -n 48 \
        --draft-max 8 --draft-min 2 --draft-p-min 0.6 \
        -p "List three Linux kernel subsystems and one job of each." \
        </dev/null > $log 2>&1
    set -l rc $status
    if test $rc -ne 0
        bad "llama-speculative rc=$rc (see $log)"
    else
        set -l tok_line (grep -E 'encoded|decoded|accept|tokens/s' $log | tail -5)
        ok "llama-speculative ran"
        for line in $tok_line
            echo "         $line"
        end
    end
end
echo

# --- 4. presenced status ---
echo "[4/4] presenced status"
if test (systemctl --user is-active presenced.service 2>/dev/null) = "active"
    # Escape the leading '-' in the pattern so fish does not try to treat
    # '-> ...' as a flag to grep.
    set -l last (journalctl --user -u presenced.service -n 200 --no-pager -o cat 2>/dev/null \
        | grep -oE '\-> (present|away_grace|away)' | tail -1 | awk '{print $2}')
    if test -z "$last"
        warn "presenced active, no FSM transitions yet (camera may be warming up)"
    else
        ok "presenced active, last FSM state: $last"
    end
else
    bad "presenced not active"
end
echo

if test $fail -eq 0
    echo "== demo PASS =="
    exit 0
else
    echo "== demo FAIL =="
    exit 1
end
