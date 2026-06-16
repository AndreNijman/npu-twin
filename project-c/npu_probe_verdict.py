#!/usr/bin/env python3
"""Print the open-stack xdna_probe verdict (one line).

Loads project-b/presenced/xdna_probe.py directly (bypassing the presenced
package __init__, which imports OpenCV etc. that aren't in the IRON venv).
Under the Phase-8 env (XRT on PATH + pyxrt importable) this prints
``npu-active-open-stack``; bare it prints ``device-only-no-userspace``.
"""
import importlib.util
import pathlib
import sys

probe_path = (
    pathlib.Path(__file__).resolve().parent.parent
    / "project-b" / "presenced" / "xdna_probe.py"
)
spec = importlib.util.spec_from_file_location("xdna_probe", probe_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["xdna_probe"] = mod
spec.loader.exec_module(mod)
print(mod.probe().verdict)
