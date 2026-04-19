from pathlib import Path

from presenced.xdna_probe import probe


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
        "xrt-only-no-inference-runtime",
        "device-only-no-userspace",
        "no-device",
    }
