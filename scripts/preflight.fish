#!/usr/bin/env fish
# Preflight checks for npu-twin on ThinkPad L16 Gen 2 (Ryzen 7 PRO 250 + Radeon 780M + XDNA 1 NPU).
# Exits 0 on pass, 1 on hard fail, 2 on warn-only (relog pending).

set -g fail 0
set -g warn 0

function check
    set -l name $argv[1]
    set -l ok $argv[2]
    if test "$ok" = 1
        echo "  [ok]   $name"
    else
        echo "  [fail] $name"
        set -g fail 1
    end
end

function soft
    set -l name $argv[1]
    set -l ok $argv[2]
    if test "$ok" = 1
        echo "  [ok]   $name"
    else
        echo "  [warn] $name"
        set -g warn 1
    end
end

echo "npu-twin preflight"
echo "=================="

set -l krel (uname -r)
set -l kmaj (string split '.' -- $krel)[1]
set -l kmin (string split '.' -- $krel)[2]
if test $kmaj -gt 6 -o \( $kmaj -eq 6 -a $kmin -ge 14 \)
    check "kernel $krel >= 6.14" 1
else
    check "kernel $krel >= 6.14" 0
end

if lspci -nn 2>/dev/null | grep -q '1022:1502'
    check "NPU PCI 1022:1502 present" 1
else
    check "NPU PCI 1022:1502 present" 0
end

if test -c /dev/accel/accel0
    check "/dev/accel/accel0" 1
else
    check "/dev/accel/accel0" 0
end

if sudo -n dmesg 2>/dev/null | grep -q 'amdxdna.*enabling device'
    check "amdxdna driver loaded (dmesg)" 1
else
    check "amdxdna driver loaded (dmesg)" 0
end

if vulkaninfo --summary 2>/dev/null | grep -q 'RADV PHOENIX'
    check "Radeon 780M RADV PHOENIX visible" 1
else
    check "Radeon 780M RADV PHOENIX visible" 0
end

if id -nG | string split ' ' | grep -q '^render$'
    check "user in 'render' group" 1
else if id -Gn $USER | string split ' ' | grep -q '^render$'
    soft "user in 'render' group (pending relog)" 0
else
    check "user in 'render' group" 0
end

if id -nG | string split ' ' | grep -q '^video$'
    check "user in 'video' group" 1
else
    check "user in 'video' group" 0
end

set -l lim (ulimit -l)
if test "$lim" = unlimited
    check "memlock unlimited" 1
else if test -f /etc/security/limits.d/99-amdxdna.conf
    soft "memlock = $lim (config present, pending relog)" 0
else
    check "memlock unlimited" 0
end

if grep -q 'amd_iommu=off' /proc/cmdline
    check "amd_iommu NOT disabled" 0
else
    check "amd_iommu NOT disabled on cmdline" 1
end

for pkg in xrt xrt-plugin-amdxdna
    if pacman -Qq $pkg >/dev/null 2>&1
        check "pacman: $pkg" 1
    else
        check "pacman: $pkg" 0
    end
end

if xrt-smi examine 2>/dev/null | grep -q 'RyzenAI-npu1'
    check "xrt-smi enumerates RyzenAI-npu1" 1
else if sudo -n xrt-smi examine 2>/dev/null | grep -q 'RyzenAI-npu1'
    soft "xrt-smi enumerates RyzenAI-npu1 (sudo-only, pending relog for unprivileged)" 0
else
    check "xrt-smi enumerates RyzenAI-npu1" 0
end

if test -c /dev/video0
    check "/dev/video0 present" 1
else
    soft "/dev/video0 missing (Project B will be gated)" 0
end

echo "=================="
if test $fail -ne 0
    echo "preflight: FAIL"
    exit 1
else if test $warn -ne 0
    echo "preflight: PASS (warnings — relog required to clear)"
    exit 0
else
    echo "preflight: PASS"
    exit 0
end
