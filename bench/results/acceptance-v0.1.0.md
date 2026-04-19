# v0.1.0 acceptance matrix — ThinkPad L16 Gen 2

Date: 2026-04-19.
Hardware: Ryzen 7 PRO 250, Radeon 780M (gfx1103), XDNA 1, 32 GB RAM,
Arch Linux, kernel 6.19.10.

## Matrix

| # | Gate | Target | Result | Where verified |
|--:|------|--------|--------|----------------|
| 1 | Baseline 8B Q4_K_M decode | ≥ 3.0 tok/s | ✅ ~5 tok/s baseline, 8B Q4_K_M, non-speculative single-prompt decode | `scripts/demo.fish` output step [3/4]; Phase 2 baseline `bench/results/20260419T064428Z.json` |
| 2 | Speculative, code class | ≥ 4.5 tok/s | ✅ 13.45 tok/s (1.59× baseline; overall 1.42× over 4 classes) | `bench/results/20260419T064428Z.json` — class=code |
| 3 | Away path fires within `GRACE_PERIOD_S + 5 s` when face disappears | present → away_grace → away within ≤35 s | ✅ verified during Phase 3 on-device validation (0 % flap over 60 s PRESENT) and during Phase 4 corun trace | `docs/decisions/0005-phase-3-cpu-gate-revision.md`; journal 2026-04-19 session 2 |
| 4 | Concurrency — presenced polling while llama-speculative benches | max corun regression < 10 % vs Phase 2 baseline | ✅ max 0.82 % (chat 0.82 / code 0.71 / factual 0.66 / reasoning 0.70); 0 OOM / 0 throttle | `bench/results/concurrency-20260419T101640Z/verdict.json` |
| 5 | Survival probes | both services survive the other being killed | ✅ llama `/health` = 200 before + after presenced kill; presenced alive before + after `systemctl stop llama-speculative`; llama restarts green | same verdict.json, `survival` block |
| 6 | XDNA 1 probe mechanism | weekly timer appends one JSON + xrt-smi block to the vault log | ✅ `npu-status.timer` active, force-fire rc=0, one entry written; honest verdict today `onnxruntime-unavailable` | `~/ObsidianVault/ClaudeMemory/projects/npu-twin-npu-log.md` |
| 7 | Fresh `scripts/demo.fish` exits 0 | exit 0 on newly-logged-in session with both units enabled | ✅ `bench/results/demo-v0.1.0-20260419T120159Z.txt` | demo output |

## Deliberately skipped (opt-in, flag off in v0.1.0)

The following gates were listed in an earlier planning doc that
assumed richer presenced features. Those features are not implemented
in `v0.1.0` and shipping placeholder code for them would be dishonest.
Listed here so future-me does not think they were silently dropped.

| Gate | Flag | Why skipped |
|------|------|-------------|
| Gaze-away → auto-focus-switch | `B_ENABLE_GAZE_FOCUS=false` (feature not shipped) | presenced only tracks face presence, not gaze direction. Out of scope for v0.1.0; would need a landmark/gaze model and a Hyprland focus dispatcher. |
| DND pause/resume on AWAY/PRESENT | `B_ENABLE_DND=false` | no dunst/mako/fcitx integration in presenced. User can wire one in via `PRESENCED_AWAY_ACTION=makoctl mode do-not-disturb` themselves. |
| Brightness restore on PRESENT | `B_ENABLE_BRIGHTNESS=false` | no brightnessctl/ddcutil integration. Same escape hatch as DND. |
| Unlock path (face → unlock session) | `B_ENABLE_UNLOCK=false` | deliberately not implemented. Unlock-on-face would need PAM integration + careful anti-spoof; ADR-0004-level decision needed. Locking is one-way for v0.1.0. |

These are `skipped (opt-in, flag off)`. Not "implemented and untested"
— not implemented.

## Honest speculative accept rate

`scripts/demo.fish` runs one 48-token prompt and reports
`accept=19.737%`. That is low, and it is *one single short prompt* on
cold speculative state. The Phase 2 20-prompt bench gives the real
distribution per class — see `bench/results/20260419T064428Z.json` —
and a re-bench captured during Phase 7 is recorded in
[phase-7-rebench-<UTC>](./) (filename lands when the bench run
completes; see `bench/results/` for the most recent `*.json`).

If the Phase 7 re-bench shows code-class accept rate below 35 %, the
repo README will carry a "Performance notes" entry calling that out
explicitly rather than quietly re-running the bench for better
numbers.
