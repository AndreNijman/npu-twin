# Phase 8 install runbook — IRON/mlir-aie/Peano on Void Linux (in-tree amdxdna)

How M1 was brought up on the ThinkPad L16 (Void, kernel 7.0.11, glibc 2.41).
Captures the non-obvious bits; most upstream docs assume Ubuntu + apt.

> **Key correction to the original Phase-8 scoping note:** the Arch-era "hard
> blocker" (in-tree driver protocol 6 vs firmware protocol 7) **does not exist
> on kernel 7.0**. The in-tree `amdxdna` here loads `npu_7.sbin`
> (`npu.sbin.1.5.5.391`, protocol 7) cleanly — **no `amdxdna-dkms`, no kernel
> module build.** Build userspace only.

## 0. Device + driver (already fine on a ≥6.14 kernel)

`amdxdna` is in-tree; `/dev/accel/accel0` exists; firmware is present at
`/lib/firmware/amdnpu/1502_00/`. Verify clean load:
```bash
sudo dmesg | grep -i amdxdna       # "Load firmware amdnpu/1502_00/npu_7.sbin" + "Initialized" — NO "Incompatible protocol"
```

## 1. Device permissions (sudo; reversible)

The accel node is `root:root 0600` by default and there is no `render` group on Void.
```bash
sudo groupadd -r render
sudo usermod -aG render andre                       # effective next login; use `sg render -c '…'` meanwhile
echo 'SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"' | sudo tee /etc/udev/rules.d/99-amdxdna.rules
printf '@render - memlock unlimited\nandre - memlock unlimited\n' | sudo tee /etc/security/limits.d/99-amdxdna.conf
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=accel --action=add
# → /dev/accel/accel0 becomes crw-rw---- root render
```

## 2. Build XRT base from amd/xdna-driver (the wheel does NOT bundle pyxrt)

The `mlir_aie` wheel ships only `aie.xrt` (an `XCLBin` helper); the IRON runtime
gates on `import pyxrt`, which comes only from a real XRT build. So XRT must be
built. Build **userspace only**.

```bash
git clone --recursive --depth 1 https://github.com/amd/xdna-driver.git ~/src/xdna-driver
# build deps (Void names):
sudo xbps-install -y boost-devel openssl-devel protobuf protobuf-devel rapidjson \
  libcurl-devel json-c-devel ncurses-devel libuuid-devel libdrm-devel elfutils-devel \
  python3-pybind11 OpenCL-Headers OpenCL-CLHPP ocl-icd-devel systemtap-devel
#   - OpenCL-Headers/CLHPP/ocl-icd-devel  → CL/cl.h (XRT base needs it even for -npu)
#   - python3-pybind11                    → pybind11Config.cmake (builds pyxrt)
#   - systemtap-devel                     → /usr/include/sys/sdt.h with STAP_PROBEV
#                                           (dtrace-utils' sdt.h is a no-macro shim — won't work)
cd ~/src/xdna-driver/xrt/build
./build.sh -npu -opt -disable-werror -j "$(nproc)"     # -disable-werror: gcc 14 + -Werror would fail
sudo make -C Release install                            # → /opt/xilinx/xrt (pyxrt, xrt-smi, libxrt_coreutil)
```

Notes:
- `build.sh` only forwards extra cmake via `-cmake-flags "<...>"`; bare `-D…` args print usage and exit.
- `pyxrt` is built against the **Python that cmake detects** (system 3.14.6 here),
  so the IRON venv must match that Python ABI (step 4 uses 3.14). If you instead
  want 3.13, do a **clean** build dir and pass `-cmake-flags
  "-DPython3_EXECUTABLE=<venv-py> -DPython3_ROOT_DIR=<…>"` — changing it on a
  warm cache does not re-detect.
- One CTest (`aie2ps_eff_net_coal_compareelf`, a Strix/XDNA2 test) fails; ignore it (108/109 pass).

## 3. Build the xdna SHIM plugin (lets XRT see /dev/accel)

```bash
# Void isn't a packaging flavor xdna-driver knows — route it to Arch's TGZ path:
sed -i 's/MATCHES "arch")/MATCHES "arch|void")/' ~/src/xdna-driver/CMake/pkg.cmake
source /opt/xilinx/xrt/setup.sh
cd ~/src/xdna-driver/build
./build.sh -release -nokmod -j "$(nproc)"               # -nokmod: do NOT touch the in-tree kernel driver
# install the plugin tree into /opt/xilinx/xrt:
tar xzf Release/xrt_plugin.*-amdxdna.tar.gz -C /tmp/xdnaplugin
sudo cp -aP /tmp/xdnaplugin/opt/xilinx/xrt/. /opt/xilinx/xrt/
# verify the device now enumerates:
sg render -c 'LD_LIBRARY_PATH=/opt/xilinx/xrt/lib64 /opt/xilinx/xrt/bin/unwrapped/xrt-smi examine'
#   → Device(s) Present: [0000:c6:00.1] RyzenAI-npu1  aie2  6x5
```

## 4. IRON toolchain (mlir-aie + Peano) — Python 3.14 to match pyxrt

```bash
uv venv --python /usr/bin/python3.14 --seed ~/src/mlir-aie/ironenv314
source ~/src/mlir-aie/ironenv314/bin/activate
pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
pip install numpy
# clone the matching examples (wheel version embeds the commit, e.g. 1.3.3.dev8+g0d49a88):
git clone --depth 1 https://github.com/Xilinx/mlir-aie.git ~/src/mlir-aie   # keep at the wheel's commit
```

`llvm-objcopy` gotcha: IRON renames a symbol in the **AIE2** kernel `.o` and
prefers `llvm-objcopy`, falling back to GNU `objcopy` — which **rejects the AIE2
machine type** (`EM_AIE 0x108`). Peano ships no `objcopy`. Install a generic one:
```bash
sudo xbps-install -y llvm21        # provides /usr/lib/llvm/21/bin/llvm-objcopy (handles any ELF e_machine)
```

## 5. Environment + run

Use `project-c/env/aie-env.sh` (do **not** `source mlir-aie/utils/env_setup.sh`
— it `return 1`s when `xrt-smi` is absent, and it would mis-set paths). It:
activates the 3.14 venv, sources `/opt/xilinx/xrt/setup.sh` (puts `pyxrt` on
`PYTHONPATH`), sets `MLIR_AIE_INSTALL_DIR`/`PEANO_INSTALL_DIR`, puts system XRT
`lib64` first on `LD_LIBRARY_PATH` (so `pyxrt` binds the matching
`libxrt_coreutil`), appends `llvm-objcopy`, and sets `NPU2=0` (Phoenix = npu1).

```bash
source project-c/env/aie-env.sh
sg render -c 'bash project-c/run/run-m1.sh'    # → PASS!
```

## System-level changes (audit / revert)

- `render` group + `andre` membership — `sudo gpasswd -d andre render; sudo groupdel render`
- `/etc/udev/rules.d/99-amdxdna.rules`, `/etc/security/limits.d/99-amdxdna.conf` — `sudo rm` + reload/relogin
- `/opt/xilinx/xrt` (XRT base + xdna plugin) — `sudo rm -rf /opt/xilinx/xrt`
- xbps packages from steps 2 & 4 (build deps, `llvm21`)
- local edit: `~/src/xdna-driver/CMake/pkg.cmake` (void→TGZ)
- **NOT changed:** in-tree `amdxdna` kernel module, NPU firmware — left exactly as shipped.
