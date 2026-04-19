#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build the trivial ONNX probe model used by `presenced --probe-npu`.

The model is a single Identity op: input `x` (1,3,64,64) float32 →
output `y` same shape. ~200 bytes. Just enough to materialise a
`onnxruntime.InferenceSession` against the VitisAIExecutionProvider
and see whether any op offloads.

Run with an environment that has `onnx` installed (stdlib doesn't have
it; pacman `python-onnx` or a venv is fine). The committed blob lives
at project-b/models/probe.onnx.

Usage:
    python -m venv /tmp/probe-venv
    /tmp/probe-venv/bin/pip install onnx
    /tmp/probe-venv/bin/python project-b/scripts/build-probe-onnx.py
"""
from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 64, 64])
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph([node], "npu_probe", [x], [y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="npu-twin",
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    here = Path(__file__).resolve().parent
    out = here.parent / "models" / "probe.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = build()
    out.write_bytes(model.SerializeToString())
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
