"""Small, dependency-light helpers shared by the edge inference scripts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def compute_letterbox(
    frame_shape: Sequence[int], input_width: int, input_height: int
) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """Return resize ratio, resized (width, height), and left/top padding."""
    frame_height, frame_width = frame_shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if input_width <= 0 or input_height <= 0:
        raise ValueError("model dimensions must be positive")

    ratio = min(input_width / frame_width, input_height / frame_height)
    resized = (
        max(1, int(round(frame_width * ratio))),
        max(1, int(round(frame_height * ratio))),
    )
    pad_x = (input_width - resized[0]) / 2
    pad_y = (input_height - resized[1]) / 2
    return ratio, resized, (int(round(pad_x - 0.1)), int(round(pad_y - 0.1)))


def letterbox(frame: np.ndarray, input_width: int, input_height: int):
    """Resize and pad a BGR frame, returning RGB float data and transform data."""
    import cv2

    ratio, resized, padding = compute_letterbox(frame.shape, input_width, input_height)
    if (frame.shape[1], frame.shape[0]) != resized:
        image = cv2.resize(frame, resized, interpolation=cv2.INTER_LINEAR)
    else:
        image = frame.copy()
    pad_x, pad_y = padding
    right = input_width - resized[0] - pad_x
    bottom = input_height - resized[1] - pad_y
    image = cv2.copyMakeBorder(
        image,
        pad_y,
        bottom,
        pad_x,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image, ratio, padding


def _prediction_matrix(raw_output: np.ndarray, label_count: int) -> np.ndarray:
    matrix = np.asarray(raw_output)
    while matrix.ndim > 2 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D YOLO output after squeezing, got {matrix.shape}")

    expected = {4 + label_count, 5 + label_count}
    if matrix.shape[0] in expected and matrix.shape[1] not in expected:
        matrix = matrix.T
    elif matrix.shape[1] not in expected and matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    if matrix.shape[1] < 5:
        raise ValueError(f"YOLO output has too few columns: {matrix.shape}")
    return matrix


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Return score-sorted indices after suppressing overlaps within each class."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    if len(boxes) == 0:
        return []

    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
            y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
            x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
            y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
                0.0, boxes[current, 3] - boxes[current, 1]
            )
            area_remaining = np.maximum(0, boxes[remaining, 2] - boxes[remaining, 0]) * np.maximum(
                0, boxes[remaining, 3] - boxes[remaining, 1]
            )
            union = area_current + area_remaining - intersection
            overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[overlap <= iou_threshold]
    return sorted(kept, key=lambda index: float(scores[index]), reverse=True)


def decode_yolo_output(
    raw_output: np.ndarray,
    frame_shape: Sequence[int],
    input_width: int,
    input_height: int,
    ratio: float,
    padding: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Decode exported YOLO boxes and map them back to the source frame."""
    if not 0 <= conf_threshold <= 1:
        raise ValueError("conf_threshold must be between 0 and 1")
    if ratio <= 0:
        raise ValueError("ratio must be positive")

    matrix = _prediction_matrix(raw_output, len(labels))
    label_count = len(labels)
    has_objectness = matrix.shape[1] == 5 + label_count
    class_start = 5 if has_objectness else 4
    class_scores = matrix[:, class_start : class_start + label_count]
    if class_scores.shape[1] == 0:
        return []
    if has_objectness:
        class_scores = class_scores * matrix[:, 4:5]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    valid = np.isfinite(scores) & (scores >= conf_threshold)
    if not np.any(valid):
        return []

    predictions = matrix[valid]
    scores = scores[valid].astype(np.float32)
    class_ids = class_ids[valid].astype(np.int64)
    centers = predictions[:, :2]
    sizes = predictions[:, 2:4]
    pad_x, pad_y = padding
    boxes = np.column_stack(
        (
            (centers[:, 0] - sizes[:, 0] / 2 - pad_x) / ratio,
            (centers[:, 1] - sizes[:, 1] / 2 - pad_y) / ratio,
            (centers[:, 0] + sizes[:, 0] / 2 - pad_x) / ratio,
            (centers[:, 1] + sizes[:, 1] / 2 - pad_y) / ratio,
        )
    )
    frame_height, frame_width = frame_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height)
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not np.any(valid_boxes):
        return []
    boxes, scores, class_ids = boxes[valid_boxes], scores[valid_boxes], class_ids[valid_boxes]
    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold)

    detections = []
    for index in keep:
        class_id = int(class_ids[index])
        detections.append(
            {
                "class_id": class_id,
                "label": labels[class_id] if class_id < len(labels) else f"ID_{class_id}",
                "confidence": float(scores[index]),
                "box": tuple(int(round(value)) for value in boxes[index]),
            }
        )
    return detections


def quantize_input(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Convert normalized float input to the dtype and scale expected by TFLite."""
    dtype = np.dtype(detail["dtype"])
    if not np.issubdtype(dtype, np.integer):
        return values.astype(dtype, copy=False)
    scale, zero_point = _quantization(detail)
    if np.any(scale <= 0):
        raise ValueError("TFLite input quantization scale must be positive")
    scale, zero_point = _broadcast_quantization(scale, zero_point, values.ndim, detail)
    quantized = np.rint(values / scale + zero_point)
    return np.clip(quantized, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)


def dequantize_output(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Convert a quantized TFLite output tensor to float values."""
    if not np.issubdtype(np.asarray(values).dtype, np.integer):
        return np.asarray(values, dtype=np.float32)
    scale, zero_point = _quantization(detail)
    scale, zero_point = _broadcast_quantization(
        scale, zero_point, np.asarray(values).ndim, detail
    )
    return (np.asarray(values, dtype=np.float32) - zero_point) * scale


def _quantization(detail: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    params = detail.get("quantization_parameters") or {}
    scales = np.asarray(params.get("scales", []), dtype=np.float32)
    zero_points = np.asarray(params.get("zero_points", []), dtype=np.float32)
    if scales.size == 0:
        scale, zero_point = detail.get("quantization", (0.0, 0))
        scales = np.asarray([scale], dtype=np.float32)
        zero_points = np.asarray([zero_point], dtype=np.float32)
    return scales, zero_points


def _broadcast_quantization(
    scales: np.ndarray, zero_points: np.ndarray, ndim: int, detail: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    if scales.size == 1:
        return scales, zero_points
    axis = int((detail.get("quantization_parameters") or {}).get("quantized_dimension", 0))
    if axis < 0 or axis >= ndim:
        raise ValueError(f"invalid TFLite quantization axis {axis} for tensor rank {ndim}")
    shape = [1] * ndim
    shape[axis] = scales.size
    return scales.reshape(shape), zero_points.reshape(shape)
