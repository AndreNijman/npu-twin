import json
from pathlib import Path

from presenced.xdna_probe import DeepProbeResult, deep_probe, probe


def test_probe_handles_missing_device(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    s = probe(device_path=missing)
    assert s.device_present is False
    assert s.verdict == "no-device"


def test_probe_verdict_is_string():
    s = probe()
    assert isinstance(s.verdict, str)
    assert s.verdict in {
        "runtime-available",
        "npu-active-open-stack",
        "xrt-only-no-inference-runtime",
        "device-only-no-userspace",
        "no-device",
    }


def test_deep_probe_returns_json_parseable_result():
    r, rc = deep_probe()
    assert isinstance(r, DeepProbeResult)
    assert rc in {0, 1, 2}
    parsed = json.loads(r.to_json())
    assert set(parsed.keys()) >= {
        "ts", "xrt_version", "fw_version", "device_present",
        "providers_tried", "providers_active",
        "offloaded_ops", "fallback_ops", "verdict",
    }
    assert parsed["verdict"] in {
        "npu-active", "cpu-fallback",
        "onnxruntime-unavailable", "vitis-ep-unavailable",
        "probe-model-missing", "session-create-failed", "inference-failed",
        "unknown",
    }


def test_deep_probe_missing_model(tmp_path: Path):
    missing = tmp_path / "nope.onnx"
    r, rc = deep_probe(model_path=missing)
    # The model check only matters if onnxruntime is importable; otherwise
    # we short-circuit to "onnxruntime-unavailable". Either verdict is fine.
    assert r.verdict in {"onnxruntime-unavailable", "probe-model-missing", "vitis-ep-unavailable"}
    assert rc == 1
