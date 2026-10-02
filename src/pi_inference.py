import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

try:
    from src.inference_utils import decode_yolo_output, dequantize_output, letterbox, quantize_input
except ModuleNotFoundError:
    from inference_utils import decode_yolo_output, dequantize_output, letterbox, quantize_input


DEFAULT_LABELS = [
    "alligator crack",
    "block crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "repair",
    "transverse crack",
]
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "best_road_anomaly_int8.tflite"


class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"Could not open video source: {src}")
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed or self.frame is None:
            self.stream.release()
            raise RuntimeError(f"Could not read an initial frame from: {src}")
        self.stopped = False
        self.lock = Lock()
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
            grabbed, frame = self.stream.read()
            if not grabbed or frame is None:
                with self.lock:
                    self.grabbed = False
                    self.stopped = True
                return
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.grabbed and self.frame is not None else None

    def stop(self):
        with self.lock:
            self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.stream.release()


class AnomalyDetector:
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

        self.interpreter = Interpreter(model_path=str(model_path), num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if len(self.input_details) != 1 or not self.output_details:
            raise ValueError("TFLite model must expose one input and at least one output")

        input_shape = tuple(int(value) for value in self.input_details[0]["shape"])
        if len(input_shape) != 4:
            raise ValueError(f"unsupported TFLite input shape: {input_shape}")
        if input_shape[-1] == 3:
            self.input_layout = "NHWC"
            self.input_height, self.input_width = input_shape[1:3]
        elif input_shape[1] == 3:
            self.input_layout = "NCHW"
            self.input_height, self.input_width = input_shape[2:4]
        else:
            raise ValueError(f"could not identify channels in TFLite input shape: {input_shape}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.labels = DEFAULT_LABELS
        if labels_path:
            with open(labels_path, "r", encoding="utf-8") as labels_file:
                self.labels = [line.strip() for line in labels_file if line.strip()]
        if not self.labels:
            raise ValueError("at least one class label is required")

        self.output_dir = Path(output_dir)
        self.detections_dir = self.output_dir / "detections"
        self.log_file = self.output_dir / "anomaly_log.csv"
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
        if self.input_layout == "NCHW":
            image = image.transpose(2, 0, 1)
        return np.expand_dims(quantize_input(image, self.input_details[0]), axis=0), ratio, padding

    def run_inference(self, frame):
        input_data, ratio, padding = self.preprocess(frame)
        input_detail = self.input_details[0]
        self.interpreter.set_tensor(input_detail["index"], input_data)
        self.interpreter.invoke()

        output_detail = self.output_details[0]
        output = dequantize_output(
            self.interpreter.get_tensor(output_detail["index"]), output_detail
        )
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


def main():
    parser = argparse.ArgumentParser(description="TFLite inference for road anomaly detection")
    parser.add_argument("--video", type=str, help="Path to a video file; omit to use the webcam")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to a TFLite model")
    parser.add_argument("--labels", type=str, help="Optional newline-separated class labels")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for class-aware NMS")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for the CSV log and snapshots")
    parser.add_argument("--event-interval", type=float, default=1.0, help="Seconds before an overlapping event can be logged again")
    parser.add_argument("--no-show", action="store_true", help="Disable the OpenCV preview window")
    args = parser.parse_args()

    source = args.video if args.video else 0
    detector = AnomalyDetector(
        args.model,
        labels_path=args.labels,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output_dir,
        event_interval=args.event_interval,
    )
    stream = VideoStream(src=source, width=640, height=480).start()
    print("Starting road anomaly detection... Press 'q' to exit.")
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    try:
        while True:
            frame = stream.read()
            if frame is None:
                if stream.stopped:
                    break
                time.sleep(0.01)
                continue

            infer_start = time.time()
            detections = detector.run_inference(frame)
            pipeline_time_ms = (time.time() - infer_start) * 1000
            pipeline_fps = 1000 / pipeline_time_ms if pipeline_time_ms > 0 else 0
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
            cv2.putText(frame, f"Video FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Pipeline FPS: {pipeline_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if not args.no_show:
                cv2.imshow("Road Anomaly Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
