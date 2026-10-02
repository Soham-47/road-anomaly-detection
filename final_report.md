# Road Anomaly Detection Report

## Scope

This project packages a YOLO11n-based road damage detector for edge-oriented inference. The repository includes PyTorch, ONNX, and TFLite model artifacts plus two inference runtimes. The checked-in dataset is not included, so training and benchmark results cannot be reproduced from this repository alone.

## Data taxonomy

Standard RDD2022 is commonly described with four road damage types: longitudinal crack, transverse crack, alligator crack, and pothole. The model artifacts in this repository expose seven labels:

- alligator crack
- block crack
- longitudinal crack
- other corruption
- pothole
- repair
- transverse crack

The additional labels are the project model's taxonomy and should not be presented as the standard four-class RDD2022 taxonomy without a documented remapping.

## Inference pipeline

Both runtimes letterbox frames to the selected model input, decode the exported YOLO output, map boxes back to the source frame, clip coordinates, and apply class-aware NMS. The TFLite runtime reads the model's input and output quantization metadata, so the checked-in INT8 model is handled with integer input tensors and dequantized outputs.

The runtimes can write CSV events and JPEG evidence. Repeated overlapping detections are suppressed with a configurable time interval. This is a cooldown and overlap filter, not a tracking system; reliable object tracking would need an explicit tracking design and evaluation.

## Artifacts and validation

The repository contains several model variants with different input resolutions. Select a model explicitly when a deployment requires a particular input size. The ONNX quantization utility uses dynamic weight quantization and prints a reminder to validate accuracy and latency on representative data.

The checked-in dataset is absent, so new INT8 TFLite exports cannot be calibrated until representative images are supplied. No edge-device FPS, model-size benefit, or accuracy improvement is claimed here without a reproducible benchmark and evaluation dataset. Before deployment, validate the selected model, class mapping, confidence thresholds, and runtime performance on the target device and footage.
