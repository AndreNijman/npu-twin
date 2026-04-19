#!/usr/bin/env fish
# Phase 4 concurrency harness.
#
# Runs the Phase 2 20-prompt speculative-decoding bench twice:
#   1. solo -- presenced stopped, baseline for this session.
#   2. corun -- presenced polling the webcam at 2 Hz during the bench.
#
# Captures amdgpu_top + top + /proc/meminfo traces for the first 60 s of
# each run, a journalctl -k window covering both runs, and a survival
# test that kills each service while the other is running.
#
# Output: bench/results/concurrency-<UTC>/
#   verdict.json        -- per-class tok/s, regression %, survival flags
#   bench-solo.json     -- copy of the solo bench result
#   bench-corun.json    -- copy of the corun bench result
#   trace-<label>/      -- amdgpu_top.json, top.log, meminfo-{start,end}.txt
#   journalctl-k.log    -- kernel log covering the full window
#   presenced*.log      -- presenced stderr/stdout for each run
#
# Acceptance (Phase 4):
#   (a) both services survive each other being killed
#   (b) corun per-class tg_speculative regression < 10 % vs Phase 2
#   (c) no OOM and no thermal-throttle events in the journal window

set -l here (status dirname)
set -g repo_root (realpath $here/../..)
set -g baseline_file "$repo_root/bench/results/20260419T064428Z.json"
set -g utc (date -u +%Y%m%dT%H%M%SZ)
set -g out_dir "$repo_root/bench/results/concurrency-$utc"
mkdir -p $out_dir
set -g t_start (date -Iseconds)

for bin in jq amdgpu_top curl top journalctl
    if not command -q $bin
        echo "error: missing $bin" >&2
        exit 1
    end
end
if test (systemctl --user is-active llama-speculative.service) != "active"
    echo "error: llama-speculative.service not active; start it first" >&2
    exit 1
end
if not test -e $baseline_file
    echo "error: missing Phase 2 baseline $baseline_file" >&2
    exit 1
end

set -q PRESENCED_YUNET_MODEL
or set -gx PRESENCED_YUNET_MODEL "$repo_root/project-b/models/yunet/face_detection_yunet_2023mar_int8.onnx"
if not test -e "$PRESENCED_YUNET_MODEL"
    echo "error: PRESENCED_YUNET_MODEL missing: $PRESENCED_YUNET_MODEL" >&2
    exit 1
end

# Neutralise presenced side-effects: no-op actions + guard the hyprctl
# dispatch by unsetting HYPRLAND_INSTANCE_SIGNATURE for the subshell.
set -gx PRESENCED_AWAY_ACTION true
set -gx PRESENCED_PRESENT_ACTION true
set -e HYPRLAND_INSTANCE_SIGNATURE

function start_trace --argument label
    set -l tdir "$out_dir/trace-$label"
    mkdir -p $tdir
    amdgpu_top -J -s 1000 -n 60 > "$tdir/amdgpu_top.json" 2>/dev/null &
    echo $last_pid > "$tdir/amdgpu_top.pid"
    top -b -d 2 -n 30 > "$tdir/top.log" 2>&1 &
    echo $last_pid > "$tdir/top.pid"
    cat /proc/meminfo > "$tdir/meminfo-start.txt"
    fish -c "sleep 60; cat /proc/meminfo > $tdir/meminfo-end.txt" &
    echo $last_pid > "$tdir/meminfo.pid"
end

function wait_trace --argument label
    set -l tdir "$out_dir/trace-$label"
    for p in $tdir/*.pid
        set -l pid (cat $p)
        wait $pid 2>/dev/null
    end
end

function run_bench --argument label
    echo ">>> bench ($label) starting at (date -Iseconds)"
    start_trace $label
    fish "$repo_root/project-a/scripts/bench.fish" > "$out_dir/bench-$label.stdout" 2>&1
    set -l rc $status
    set -l latest (command ls -t $repo_root/bench/results/*.json | head -1)
    cp $latest "$out_dir/bench-$label.json"
    wait_trace $label
    echo "<<< bench ($label) done rc=$rc, copied $latest -> bench-$label.json"
    return $rc
end

echo "Phase 4 concurrency harness -- $utc"
echo "out: $out_dir"
echo

# --- 1. solo run (presenced OFF) ---
pkill -f "python -m presenced" 2>/dev/null
sleep 1
run_bench solo

# --- 2. corun (presenced ON) ---
python -m presenced > "$out_dir/presenced-corun.log" 2>&1 &
set -g presenced_pid $last_pid
sleep 3
if not kill -0 $presenced_pid 2>/dev/null
    echo "error: presenced died immediately; see $out_dir/presenced-corun.log" >&2
    exit 1
end
run_bench corun
kill $presenced_pid 2>/dev/null
wait $presenced_pid 2>/dev/null

# --- 3. survival test ---
echo ">>> survival test"
python -m presenced > "$out_dir/presenced-survive.log" 2>&1 &
set -g p_pid $last_pid
sleep 3
set -l llama_before (curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:11434/health)
kill $p_pid 2>/dev/null
wait $p_pid 2>/dev/null
sleep 2
set -l llama_after_p_kill (curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:11434/health)

# Restart presenced, then stop/restart llama-speculative
python -m presenced > "$out_dir/presenced-survive2.log" 2>&1 &
set -g p_pid2 $last_pid
sleep 3
set -l p_alive_pre 1
kill -0 $p_pid2 2>/dev/null; and set p_alive_pre 1; or set p_alive_pre 0
systemctl --user stop llama-speculative.service
sleep 3
set -l p_alive_post 0
kill -0 $p_pid2 2>/dev/null; and set p_alive_post 1; or set p_alive_post 0
systemctl --user start llama-speculative.service
sleep 5
set -l llama_restarted (curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:11434/health)
kill $p_pid2 2>/dev/null
wait $p_pid2 2>/dev/null

# --- 4. kernel log + dmesg window ---
journalctl -k -S "$t_start" > "$out_dir/journalctl-k.log" 2>&1
set -l oom_hits (grep -cE 'Out of memory|oom_kill|invoked oom-killer' "$out_dir/journalctl-k.log")
set -l throttle_hits (grep -cE 'thermal.*throttl|cpufreq.*throttl|throttl' "$out_dir/journalctl-k.log")

# --- 5. verdict ---
echo ">>> writing verdict"
jq -n \
    --arg utc "$utc" \
    --arg t_start "$t_start" \
    --slurpfile baseline $baseline_file \
    --slurpfile solo "$out_dir/bench-solo.json" \
    --slurpfile corun "$out_dir/bench-corun.json" \
    --argjson llama_before "$llama_before" \
    --argjson llama_after_p_kill "$llama_after_p_kill" \
    --argjson p_alive_pre "$p_alive_pre" \
    --argjson p_alive_post "$p_alive_post" \
    --argjson llama_restarted "$llama_restarted" \
    --argjson oom_hits "$oom_hits" \
    --argjson throttle_hits "$throttle_hits" \
    '
      def cls:
        group_by(.class)
        | map({
            class:.[0].class,
            n:length,
            avg_spec: ([.[].tg_speculative | numbers] | if length>0 then add/length else null end),
            avg_accept: ([.[].accept_rate | numbers] | if length>0 then add/length else null end)
          });
      def regression(b; a):
        if (b == null or a == null or b == 0) then null
        else ((b - a) / b * 100) end;

      ($baseline[0].results | cls) as $b
      | ($solo[0].results   | cls) as $s
      | ($corun[0].results  | cls) as $c
      | [$b[] as $bi
         | ($s[] | select(.class == $bi.class)) as $si
         | ($c[] | select(.class == $bi.class)) as $ci
         | {
             class: $bi.class,
             baseline_spec: $bi.avg_spec,
             solo_spec: $si.avg_spec,
             corun_spec: $ci.avg_spec,
             regression_pct_vs_baseline: regression($bi.avg_spec; $ci.avg_spec),
             regression_pct_vs_solo:     regression($si.avg_spec; $ci.avg_spec)
           }] as $per_class
      | {
          utc: $utc,
          t_start: $t_start,
          per_class: $per_class,
          max_regression_vs_baseline_pct: ([$per_class[].regression_pct_vs_baseline | numbers] | max),
          max_regression_vs_solo_pct:     ([$per_class[].regression_pct_vs_solo     | numbers] | max),
          survival: {
            llama_health_before: $llama_before,
            llama_health_after_presenced_killed: $llama_after_p_kill,
            presenced_alive_before_llama_stop: ($p_alive_pre == 1),
            presenced_alive_after_llama_stop: ($p_alive_post == 1),
            llama_restarted_ok: $llama_restarted
          },
          kernel: {
            oom_events: $oom_hits,
            throttle_events: $throttle_hits
          }
        }
    ' > "$out_dir/verdict.json"

cat "$out_dir/verdict.json" | jq .
echo
echo "wrote $out_dir/verdict.json"
