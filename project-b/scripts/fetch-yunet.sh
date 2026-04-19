#!/usr/bin/env bash
# Fetch YuNet face-detection ONNX models from OpenCV Zoo.
#
# Pulls two artifacts:
#   - face_detection_yunet_2023mar.onnx        (FP32 reference)
#   - face_detection_yunet_2023mar_int8.onnx   (INT8 quantized)
#
# The INT8 model is produced upstream by OpenCV Zoo's own quantize.py
# (see opencv_zoo/tools/quantize/), calibrated on a WIDER FACE subset.
# We vendor the pre-quantized artifact rather than re-running calibration
# locally: the upstream output is reproducible, signed by the zoo release,
# and avoids pulling ppq + a calibration dataset into this repo.
#
# If a future need arises to re-quantize with site-specific calibration
# (e.g. this laptop's IR camera under low light), run opencv_zoo's
# tools/quantize/quantize-ort.py against face_detection_yunet_2023mar.onnx
# with local calibration frames and drop the result at
# project-b/models/yunet/face_detection_yunet_2023mar_int8.onnx. The
# runtime contract is byte-for-byte compatible.
#
# Usage:
#   project-b/scripts/fetch-yunet.sh
#
# Env overrides:
#   PRESENCED_MODEL_DIR  destination dir (default: project-b/models/yunet)
#   YUNET_TAG            opencv_zoo git tag (default: main)
#   CURL                 curl binary (default: curl)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/.." && pwd)"
DEFAULT_DIR="$PROJECT_DIR/models/yunet"
MODEL_DIR="${PRESENCED_MODEL_DIR:-$DEFAULT_DIR}"
TAG="${YUNET_TAG:-main}"
CURL="${CURL:-curl}"

BASE="https://github.com/opencv/opencv_zoo/raw/${TAG}/models/face_detection_yunet"
FILES=(
  "face_detection_yunet_2023mar.onnx"
  "face_detection_yunet_2023mar_int8.onnx"
)

mkdir -p "$MODEL_DIR"

for f in "${FILES[@]}"; do
  dest="$MODEL_DIR/$f"
  if [[ -s "$dest" ]]; then
    echo "exists: $dest"
    continue
  fi
  url="$BASE/$f"
  echo "fetch:  $url"
  tmp="$(mktemp "${dest}.XXXX")"
  if ! "$CURL" -L --fail --silent --show-error -o "$tmp" "$url"; then
    rm -f "$tmp"
    echo "error: download failed for $url" >&2
    exit 1
  fi
  mv "$tmp" "$dest"
done

echo
echo "saved to: $MODEL_DIR"
echo
echo "sha256 (record these in your own log if you vendor):"
sha256sum "${FILES[@]/#/$MODEL_DIR/}"
echo
echo "point presenced at the INT8 model with:"
echo "  PRESENCED_YUNET_MODEL=$MODEL_DIR/face_detection_yunet_2023mar_int8.onnx"
