import numpy as np

from src.inference_utils import (
    compute_letterbox,
    decode_yolo_output,
    dequantize_output,
    quantize_input,
)


LABELS = ["alligator crack", "block crack", "longitudinal crack", "other corruption", "pothole", "repair", "transverse crack"]


def test_compute_letterbox_preserves_aspect_ratio():
    ratio, resized, padding = compute_letterbox((480, 640), 320, 320)

    assert ratio == 0.5
    assert resized == (320, 240)
    assert padding == (0, 40)


def test_decode_uses_class_aware_nms_and_clips_boxes():
    raw = np.zeros((1, 11, 3), dtype=np.float32)
    raw[0, :4, :] = np.array(
        [[50, 50, 50], [50, 50, 50], [60, 60, 60], [60, 60, 60]]
    )
    raw[0, 4, :] = [0.9, 0.8, 0.0]
    raw[0, 5, 2] = 0.85
    raw[0, :4, 2] = [0, 0, 60, 60]

    detections = decode_yolo_output(
        raw,
        frame_shape=(100, 100, 3),
        input_width=100,
        input_height=100,
        ratio=1.0,
        padding=(0, 0),
        conf_threshold=0.25,
        iou_threshold=0.45,
        labels=LABELS,
    )

    assert len(detections) == 2
    assert {d["class_id"] for d in detections} == {0, 1}
    assert detections[0]["box"] == (20, 20, 80, 80)
    assert detections[1]["box"] == (0, 0, 30, 30)


def test_quantization_uses_tflite_scale_and_zero_point():
    detail = {
        "dtype": np.int8,
        "quantization": (0.1, -128),
        "quantization_parameters": {"scales": np.array([]), "zero_points": np.array([])},
    }
    values = np.array([[[[0.0, 0.5, 1.0]]]], dtype=np.float32)

    quantized = quantize_input(values, detail)

    assert quantized.tolist() == [[[[-128, -123, -118]]]]
    np.testing.assert_allclose(dequantize_output(quantized, detail), values)
