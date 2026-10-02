# Road Anomaly Detection

This repository contains YOLO-based road damage detection runtimes and the exported model artifacts used by them. The deployment scripts support TFLite on Raspberry Pi class devices and ONNX Runtime on CPU systems.

## Repository contents

```text
data/yolo_format/data.yaml       Dataset layout and seven class names
models/                           PyTorch, ONNX, and TFLite model artifacts
requirements-pi.txt               Runtime-only Raspberry Pi dependencies
requirements.txt                  Full export and development dependencies
scripts/export_tflite.py          Export a YOLO model to TFLite
scripts/quantize_onnx.py          Dynamic ONNX weight quantization
src/pi_inference.py               TFLite camera or video inference
src/onnx_inference.py             ONNX video inference
final_report.md                   Project report and scope notes
```

The image and label files referenced by `data/yolo_format/data.yaml` are not included. The config leaves `path` unset for portability; `scripts/export_tflite.py` creates a temporary config with the repository's absolute dataset path. To export INT8 or train locally, place the dataset under `data/yolo_format/images/{train,val,test}` and the matching label directories, or update the dataset root for your own command. The model files in `models/` expose seven labels used by this project:

`alligator crack`, `block crack`, `longitudinal crack`, `other corruption`, `pothole`, `repair`, and `transverse crack`.

## Install

For Raspberry Pi inference, install only the runtime dependencies:

```bash
python3 -m pip install -r requirements-pi.txt
```

`tflite-runtime` is installed by the Pi requirements file only on ARM platforms. On x86 or other platforms where that wheel is unavailable, install a compatible TensorFlow package separately before using the TFLite runtime; the script then uses TensorFlow's interpreter fallback. For model export, ONNX inference, or development tooling, install the full `requirements.txt` instead.

## Run TFLite inference

The Pi runtime defaults to the checked-in INT8 model. It letterboxes frames to the model input, dequantizes model outputs, applies class-aware NMS, and writes deduplicated events to `anomaly_log.csv` and `detections/`.

```bash
python3 src/pi_inference.py --video path/to/video.mp4 --no-show
```

Use `--model` to select another TFLite artifact, `--output-dir` to choose the log and snapshot directory, and `--event-interval` to adjust repeated-event suppression. The event filter is a cooldown and overlap check; it is not object tracking.

## Run ONNX inference

```bash
python3 src/onnx_inference.py \
  --video path/to/video.mp4 \
  --model models/best_road_anomaly.onnx \
  --no-show
```

Add `--save --output path/to/output.mp4` to write an annotated video. The runtime validates the input video and output writer before processing. It does not claim a measured edge FPS; benchmark a target device with representative footage before deployment.

Both runtime previews label the measured value `Pipeline FPS`; it includes preprocessing, runtime inference, output decoding, and event logging or snapshot writes, so it is not a model-only benchmark.

## Export and quantize models

These commands require the full `requirements.txt` dependencies and a source model. Existing output files are preserved unless `--force` is supplied. The checked-in dataset is absent, so the default INT8 export cannot run until calibration images are added. Float32 export does not require INT8 calibration data.

```bash
python3 scripts/export_tflite.py --model models/best_road_anomaly.pt --float32 --force
# After adding calibration images under data/yolo_format:
python3 scripts/export_tflite.py --model models/best_road_anomaly.pt --force
python3 scripts/quantize_onnx.py --input models/best_road_anomaly.onnx
```

The ONNX script performs dynamic weight quantization. Measure accuracy and latency on representative data before using a quantized model in production.
