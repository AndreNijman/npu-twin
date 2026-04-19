# ADR-0006: Retire Haar skeleton detector after YuNet validation

Date: 2026-04-19
Status: Accepted
Related: [ADR-0004](0004-yunet-over-haar.md), [ADR-0005](0005-phase-3-cpu-gate-revision.md)

## Context

[ADR-0004](0004-yunet-over-haar.md) swapped the primary face detector
from OpenCV Haar to YuNet ONNX but kept Haar reachable via
`PRESENCED_DETECTOR=haar` for **one commit cycle** as a revert knob,
in case YuNet surprised us on real hardware.

Phase 3 on-device validation (2026-04-19, see
[ADR-0005](0005-phase-3-cpu-gate-revision.md) for numbers) cleared
YuNet on every gate that measures detection quality:

- FSM flap rate: **0 %** (0 transitions over two 60 s PRESENT windows,
  120 samples each).
- `detect()` p50 / p95: **7.78 ms / 9.59 ms** (n=200, 100 % hit rate).
- Detector swap round-trip (`yunet → haar → yunet`): both boot clean,
  FSM transitions fire on both. Parity confirmed.
- `pytest -q`: 17 / 17 green.

No surprises. No observed failure mode that would require a Haar
fallback.

## Decision

Remove Haar from `presenced`:

- Delete `OpenCVHaar`, `_resolve_haar_cascade`, `_HAAR_FILENAME`,
  `_HAAR_SEARCH_PATHS` from `project-b/presenced/face.py`.
- Remove the `"haar"` arm from `build()` — unknown backend values now
  raise with `expected 'yunet'` only.
- Remove `PRESENCED_HAAR_CASCADE` from `project-b/presenced/config.py`
  and the env var table.
- Delete `project-b/tests/test_face_haar.py` and any Haar-only
  fixtures from the shared test helpers.
- Keep the `detector` config key so future backends (MediaPipe,
  ONNX-backed alternatives) can slot in without touching the config
  schema.

## Consequences

- One-line rollback to Haar is gone. If YuNet regresses in the field
  a future fix ships a new backend, not a `PRESENCED_DETECTOR=haar`
  flip.
- No runtime cost — the system-path Haar XML on Arch
  (`/usr/share/opencv4/haarcascades/…`) stays installed with the
  `opencv` package but `presenced` stops referencing it.
- Removes the `cv2.data` fragility story from the detector module
  entirely; the YuNet path never touched `cv2.data` to begin with.
  The `errors-and-fixes/cv2-data-missing-on-arch` vault note becomes
  historical.
