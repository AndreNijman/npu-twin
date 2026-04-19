#!/usr/bin/env fish
# check-npu-status.fish — weekly XDNA 1 status probe.
#
# Runs `xrt-smi examine` + `presenced --probe-npu`, appends one dated
# record to the vault log at
#   ~/ObsidianVault/ClaudeMemory/projects/npu-twin-npu-log.md
#
# Fired weekly by project-b/contrib/systemd/npu-status.timer. Safe to
# run by hand.

set -l here (status dirname)
set -g repo_root (realpath $here/..)
set -g vault_log "$HOME/ObsidianVault/ClaudeMemory/projects/npu-twin-npu-log.md"
set -g utc (date -u +%Y-%m-%dT%H:%M:%SZ)

if not test -f $vault_log
    echo "error: vault log missing at $vault_log" >&2
    exit 1
end

set -l xrt_out (xrt-smi examine 2>&1)
set -l xrt_rc $status

set -l probe_json ""
set -l probe_rc 99
if test -f "$repo_root/project-b/presenced/__main__.py"
    pushd $repo_root/project-b >/dev/null
    set probe_json (env PYTHONPATH=. python -m presenced --probe-npu 2>/dev/null)
    set probe_rc $status
    popd >/dev/null
else
    set probe_json '{"error":"presenced package not found"}'
end

set -l verdict (echo $probe_json | jq -r '.verdict // "parse-error"' 2>/dev/null)
test -z "$verdict"; and set verdict "parse-error"

begin
    echo ""
    echo "## $utc"
    echo ""
    echo "- **host:** "(hostname)
    echo "- **probe rc:** $probe_rc"
    echo "- **probe verdict:** $verdict"
    echo "- **xrt-smi rc:** $xrt_rc"
    echo ""
    echo "### presenced --probe-npu"
    echo ""
    echo '```json'
    # Pipe directly so jq's newlines survive; fall back to raw JSON if
    # jq can't parse it.
    echo $probe_json | jq . 2>/dev/null
    if test $pipestatus[-1] -ne 0
        echo $probe_json
    end
    echo '```'
    echo ""
    echo "### xrt-smi examine (head -20)"
    echo ""
    echo '```'
    printf '%s\n' $xrt_out | head -20
    echo '```'
    echo ""
    echo "---"
end >> $vault_log

echo "wrote entry for $utc to $vault_log (probe rc=$probe_rc, verdict=$verdict)"
exit 0
