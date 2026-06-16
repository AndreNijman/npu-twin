#!/usr/bin/env fish
# check-npu-status.fish — XDNA 1 NPU status probe (Phase 8-aware).
#
# Appends one dated record to a REPO-LOCAL log:
#   project-c/npu-status-log.md
# (the old ~/ObsidianVault/ClaudeMemory path is dead post Arch->Void migration;
#  the vault now lives behind the claude-memory MCP, not a local file.)
#
# Records `xrt-smi examine` + the open-stack xdna_probe verdict. Since Phase 8
# the open IRON/mlir-aie/Peano stack drives the NPU, so the verdict reads
# `npu-active-open-stack` when the toolchain (XRT + pyxrt) is on PATH/PYTHONPATH.
#
# Fired weekly by project-b/contrib/systemd/npu-status.timer. Safe to run by hand.
# The xrt-smi device query needs the caller in the `render` group (else run via
# `sg render -c '...'`). The probe verdict needs the Phase-8 env sourced — point
# AIE_ENV at it (default ~/src/mlir-aie/aie-env314.sh).

set -l here (status dirname)
set -g repo_root (realpath $here/..)
set -g log_file "$repo_root/project-c/npu-status-log.md"
set -g utc (date -u +%Y-%m-%dT%H:%M:%SZ)

# Overridable locations.
set -q XRT_PREFIX; or set -g XRT_PREFIX /opt/xilinx/xrt
set -q AIE_ENV; or set -g AIE_ENV "$HOME/src/mlir-aie/aie-env314.sh"
set -l xrt_smi "$XRT_PREFIX/bin/unwrapped/xrt-smi"

if not test -f $log_file
    printf '# npu-twin — NPU status log\n\nAppend-only. One record per run of `scripts/check-npu-status.fish`.\nSince Phase 8 the open IRON/mlir-aie/Peano stack drives the NPU, so `verdict`\nreads `npu-active-open-stack` once the toolchain is installed (see project-c/).\n\n<!-- entries below — check-npu-status.fish appends after this marker -->\n' > $log_file
end

# xrt-smi examine (open-stack XRT built from amd/xdna-driver).
set -l xrt_out "xrt-smi not found at $xrt_smi"
set -l xrt_rc 127
if test -x $xrt_smi
    set xrt_out (env LD_LIBRARY_PATH="$XRT_PREFIX/lib64" $xrt_smi examine 2>&1)
    set xrt_rc $status
end

# Open-stack verdict via xdna_probe.probe() under the Phase-8 env.
set -l verdict "unknown"
if test -f $AIE_ENV
    set verdict (bash -c "source '$AIE_ENV' >/dev/null 2>&1; python '$repo_root/project-c/npu_probe_verdict.py' 2>/dev/null")
else
    set verdict (bash -c "python '$repo_root/project-c/npu_probe_verdict.py' 2>/dev/null")
end
test -z "$verdict"; and set verdict "probe-error"

begin
    echo ""
    echo "## $utc"
    echo ""
    echo "- **host:** "(hostname)
    echo "- **verdict:** $verdict"
    echo "- **xrt-smi rc:** $xrt_rc"
    echo ""
    echo "### xrt-smi examine (head -24)"
    echo ""
    echo '```'
    printf '%s\n' $xrt_out | head -24
    echo '```'
    echo ""
    echo "---"
end >> $log_file

echo "wrote entry for $utc -> $log_file (verdict=$verdict, xrt-smi rc=$xrt_rc)"
exit 0
