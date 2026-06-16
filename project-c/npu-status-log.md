# npu-twin — NPU status log

Append-only. One record per run of `scripts/check-npu-status.fish`.
Since Phase 8 the open IRON/mlir-aie/Peano stack drives the NPU, so `verdict`
reads `npu-active-open-stack` once the toolchain is installed (see project-c/).

<!-- entries below — check-npu-status.fish appends after this marker -->

## 2026-06-16T14:03:28Z

- **host:** AndreVoid
- **verdict:** npu-active-open-stack
- **xrt-smi rc:** 0

### xrt-smi examine (head -24)

```
System Configuration
  OS Name              : Linux
  Release              : 7.0.11_1
  Machine              : x86_64
  CPU Cores            : 16
  Memory               : 30701 MB
  Distribution         : Void Linux
  GLIBC                : 2.41
  Model                : 21SCCTO1WW
  BIOS Vendor          : LENOVO
  BIOS Version         : R2UET27W (1.27 )
  Processor            : AMD Ryzen 7 PRO 250 w/ Radeon 780M Graphics

XRT
  Version              : 2.25.0
  Branch               : HEAD
  Hash                 : 943586a79b5a714463cb13d3ba7e178b8532c817
  Hash Date            : Tue, 9 Jun 2026 19:31:23 -0700
  amdxdna Version      : 7.0.11_1
  virtio-pci Version   : 7.0.11_1
  NPU Firmware Version : 1.5.5.391

Device(s) Present
|BDF             |Name          |Architecture  |Topology  |
```

---
