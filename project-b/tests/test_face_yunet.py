import numpy as np
import pytest

from presenced.face import YuNet, _resolve_yunet_model


def _frame(w: int = 640, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class _FakeCap:
    def __init__(self, frames, opened: bool = True):
        self._frames = list(frames)
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


class _FakeYN:
    def __init__(self, faces_seq):
        self._faces = list(faces_seq)
        self.size_calls: list[tuple[int, int]] = []

    def setInputSize(self, wh):
        self.size_calls.append(tuple(wh))

    def detect(self, _frame):
        if not self._faces:
            return 0, None
        return 1, self._faces.pop(0)


def _patch_cv2(monkeypatch, cap, yn):
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", lambda _idx: cap)

    class _Factory:
        @staticmethod
        def create(**_kwargs):
            return yn

    monkeypatch.setattr(cv2, "FaceDetectorYN", _Factory)


def _model_env(monkeypatch, tmp_path):
    p = tmp_path / "yunet.onnx"
    p.write_bytes(b"\x00")
    monkeypatch.setenv("PRESENCED_YUNET_MODEL", str(p))
    return p


def test_resolve_yunet_model_missing_env(monkeypatch):
    monkeypatch.delenv("PRESENCED_YUNET_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="PRESENCED_YUNET_MODEL is not set"):
        _resolve_yunet_model()


def test_resolve_yunet_model_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("PRESENCED_YUNET_MODEL", str(tmp_path / "nope.onnx"))
    with pytest.raises(RuntimeError, match="does not exist"):
        _resolve_yunet_model()


def test_resolve_yunet_model_ok(monkeypatch, tmp_path):
    p = _model_env(monkeypatch, tmp_path)
    assert _resolve_yunet_model() == str(p)


def test_yunet_returns_true_when_faces_detected(monkeypatch, tmp_path):
    _model_env(monkeypatch, tmp_path)
    cap = _FakeCap([_frame(), _frame()])
    yn = _FakeYN([np.ones((1, 15), dtype=np.float32)])
    _patch_cv2(monkeypatch, cap, yn)
    d = YuNet()
    assert d.detect() is True
    assert yn.size_calls == [(640, 480)]
    d.close()
    assert cap.released


def test_yunet_returns_false_when_no_faces(monkeypatch, tmp_path):
    _model_env(monkeypatch, tmp_path)
    cap = _FakeCap([_frame(), _frame()])
    yn = _FakeYN([None])
    _patch_cv2(monkeypatch, cap, yn)
    d = YuNet()
    assert d.detect() is False
    d.close()


def test_yunet_returns_false_when_faces_empty_array(monkeypatch, tmp_path):
    _model_env(monkeypatch, tmp_path)
    cap = _FakeCap([_frame(), _frame()])
    yn = _FakeYN([np.zeros((0, 15), dtype=np.float32)])
    _patch_cv2(monkeypatch, cap, yn)
    d = YuNet()
    assert d.detect() is False
    d.close()


def test_yunet_returns_false_on_read_failure(monkeypatch, tmp_path):
    _model_env(monkeypatch, tmp_path)
    # Only one frame available (the warm-up). The real detect() call will see
    # an empty frame queue and receive ok=False.
    cap = _FakeCap([_frame()])
    yn = _FakeYN([np.ones((1, 15), dtype=np.float32)])
    _patch_cv2(monkeypatch, cap, yn)
    d = YuNet()
    assert d.detect() is False
    d.close()


def test_yunet_resizes_detector_on_frame_size_change(monkeypatch, tmp_path):
    _model_env(monkeypatch, tmp_path)
    cap = _FakeCap([_frame(640, 480), _frame(640, 480), _frame(1280, 720)])
    yn = _FakeYN(
        [
            np.ones((1, 15), dtype=np.float32),
            np.ones((1, 15), dtype=np.float32),
        ]
    )
    _patch_cv2(monkeypatch, cap, yn)
    d = YuNet()
    assert d.detect() is True
    assert d.detect() is True
    assert yn.size_calls == [(640, 480), (1280, 720)]
    d.close()


def test_yunet_missing_camera_raises(monkeypatch, tmp_path):
    _model_env(monkeypatch, tmp_path)
    cap = _FakeCap([], opened=False)
    yn = _FakeYN([])
    _patch_cv2(monkeypatch, cap, yn)
    d = YuNet(camera_index=7)
    with pytest.raises(RuntimeError, match="cannot open camera index 7"):
        d.detect()
