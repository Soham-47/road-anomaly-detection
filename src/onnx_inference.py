import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

import cv2
import numpy as np
import onnxruntime as ort

try:
    from src.inference_utils import decode_yolo_output, letterbox
except ModuleNotFoundError:
    from inference_utils import decode_yolo_output, letterbox


DEFAULT_LABELS = [
    "alligator crack",
    "block crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "repair",
    "transverse crack",
]
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "best_road_anomaly.onnx"


class ONNXAnomalyDetector:
    def __init__(
        self,
        model_path,
        labels_path=None,
        conf_threshold=0.25,
        iou_threshold=0.45,
        output_dir=".",
        event_interval=1.0,
    ):
        if not 0 <= conf_threshold <= 1:
            raise ValueError("conf_threshold must be between 0 and 1")
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if event_interval < 0:
            raise ValueError("event_interval must be non-negative")

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise ValueError("ONNX model must expose one input and at least one output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        input_shape = inputs[0].shape
        if len(input_shape) != 4 or not all(isinstance(value, (int, np.integer)) for value in input_shape[2:]):
            raise ValueError(f"ONNX model must have a static image input, got {input_shape}")
        self.input_height, self.input_width = int(input_shape[2]), int(input_shape[3])
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError(f"invalid ONNX input dimensions: {input_shape}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.labels = DEFAULT_LABELS
        if labels_path:
            with open(labels_path, "r", encoding="utf-8") as labels_file:
                self.labels = [line.strip() for line in labels_file if line.strip()]
        if not self.labels:
            raise ValueError("at least one class label is required")

        self.output_dir = Path(output_dir)
        self.detections_dir = self.output_dir / "onnx_detections"
        self.log_file = self.output_dir / "onnx_anomaly_log.csv"
        self.detections_dir.mkdir(parents=True, exist_ok=True)
        self.event_interval = event_interval
        self._last_events = []
        self._init_logger()

    def _init_logger(self):
        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as log_file:
                csv.writer(log_file).writerow(["Timestamp", "AnomalyType", "Confidence"])

    def log_anomaly(self, anomaly_type, confidence):
        with self.log_file.open("a", newline="", encoding="utf-8") as log_file:
            csv.writer(log_file).writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), anomaly_type, float(confidence)]
            )

    def preprocess(self, frame):
        image, ratio, padding = letterbox(frame, self.input_width, self.input_height)
        return image.transpose(2, 0, 1)[None, ...], ratio, padding

    def run_inference(self, frame):
        input_data, ratio, padding = self.preprocess(frame)
        output = self.session.run([self.output_name], {self.input_name: input_data})[0]
        detections = decode_yolo_output(
            output,
            frame.shape,
            self.input_width,
            self.input_height,
            ratio,
            padding,
            self.conf_threshold,
            self.iou_threshold,
            self.labels,
        )
        for detection in detections:
            if self._event_allowed(detection):
                self.log_anomaly(detection["label"], detection["confidence"])
                self.save_anomaly_snapshot(frame, detection["label"], detection["box"])
        return detections

    def _event_allowed(self, detection):
        now = time.monotonic()
        self._last_events = [
            event for event in self._last_events if now - event["time"] < self.event_interval
        ]
        for event in self._last_events:
            if event["label"] == detection["label"] and _box_iou(event["box"], detection["box"]) >= 0.5:
                return False
        self._last_events.append({"time": now, "label": detection["label"], "box": detection["box"]})
        return True

    def save_anomaly_snapshot(self, frame, label, box):
        safe_label = "".join(character if character.isalnum() or character in "-_." else "_" for character in label)
        filename = self.detections_dir / f"{safe_label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        snapshot = frame.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(snapshot, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if not cv2.imwrite(str(filename), snapshot):
            raise RuntimeError(f"Could not write detection snapshot: {filename}")


def _box_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class AsyncInference:
    def __init__(self, detector):
        self.detector = detector
        self.frame = None
        self.frame_sequence = 0
        self.result_sequence = -1
        self.result_frame = None
        self.detections = []
        self.stopped = False
        self.lock = Lock()
        self.latency = 0
        self.thread = None

    def start(self):
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while True:
            with self.lock:
                if self.stopped:
                    return
                if self.frame is None:
                    local_frame = None
                else:
                    local_frame = self.frame.copy()
                    local_sequence = self.frame_sequence
            if local_frame is None:
                time.sleep(0.01)
                continue

            start = time.time()
            new_detections = self.detector.run_inference(local_frame)
            with self.lock:
                self.detections = new_detections
                self.latency = (time.time() - start) * 1000
                self.result_sequence = local_sequence
                self.result_frame = local_frame
                # A newer frame may have arrived while inference was running.
                if self.frame_sequence == local_sequence:
                    self.frame = None

    def set_frame(self, frame):
        with self.lock:
            self.frame_sequence += 1
            self.frame = frame.copy()
            return self.frame_sequence

    def get_detections(self, sequence=None):
        with self.lock:
            if sequence is not None and sequence != self.result_sequence:
                return [], self.latency
            return list(self.detections), self.latency

    def get_result(self, last_sequence=None):
        """Return each completed result with the frame that produced it."""
        with self.lock:
            if self.result_frame is None or self.result_sequence == last_sequence:
                return None, [], self.latency, self.result_sequence
            return (
                self.result_frame.copy(),
                list(self.detections),
                self.latency,
                self.result_sequence,
            )

    def stop(self):
        with self.lock:
            self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)


def _open_writer(path, fps, size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "XVID"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open video writer: {path}")


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for road anomaly detection")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to an ONNX model")
    parser.add_argument("--labels", type=str, help="Optional newline-separated class labels")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for logs and snapshots")
    parser.add_argument("--event-interval", type=float, default=1.0, help="Seconds before an overlapping event can be logged again")
    parser.add_argument("--save", action="store_true", help="Save the annotated output video")
    parser.add_argument("--output", type=str, default="output_detection.mp4", help="Output video path")
    parser.add_argument("--show", dest="show", action="store_true", default=True, help="Display the video")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable the preview window")
    parser.add_argument("--loop", action="store_true", help="Loop the video")
    args = parser.parse_args()

    detector = ONNXAnomalyDetector(
        args.model,
        labels_path=args.labels,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output_dir,
        event_interval=args.event_interval,
    )
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120:
        video_fps = 30
    frame_delay = max(1, int(1000 / video_fps))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Input video has invalid dimensions")

    out = None
    async_infer = None
    last_result_sequence = -1
    try:
        out = _open_writer(args.output, video_fps, (width, height)) if args.save else None
        async_infer = AsyncInference(detector).start()
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.loop and cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    continue
                break

            async_infer.set_frame(frame)
            result_frame, detections, latency, result_sequence = async_infer.get_result(
                last_result_sequence
            )
            if result_frame is not None:
                frame = result_frame
                last_result_sequence = result_sequence
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detection['label']} {detection['confidence']:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
            pipeline_fps = 1000 / latency if latency > 0 else 0
            cv2.putText(frame, f"Video FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pipeline FPS: {pipeline_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if out is not None:
                out.write(frame)
            if args.show:
                cv2.imshow("Async Road Anomaly Detection", frame)
                if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
                    break
    finally:
        if async_infer is not None:
            async_infer.stop()
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
