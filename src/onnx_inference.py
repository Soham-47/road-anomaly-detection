import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime
import onnxruntime as ort
import argparse
from threading import Thread, Lock

class ONNXAnomalyDetector:
    def __init__(self, model_path, labels_path=None, conf_threshold=0.25, iou_threshold=0.45):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.labels = ["alligator crack", "block crack", "longitudinal crack", "other corruption", "pothole", "repair", "transverse crack"]
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        self.log_file = "onnx_anomaly_log.csv"
        self._init_logger()

    def _init_logger(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "AnomalyType", "Confidence"])

    def log_anomaly(self, anomaly_type, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, anomaly_type, confidence])

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        r = min(self.input_width / w, self.input_height / h)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = self.input_width - new_unpad[0], self.input_height - new_unpad[1]
        dw /= 2
        dh /= 2
        if (w, h) != new_unpad:
            img = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img = frame.copy()
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img, r, (left, top)

    def nms(self, boxes, scores, iou_threshold):
        if len(boxes) == 0: return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= iou_threshold)[0] + 1]
        return keep

    def run_inference(self, frame):
        input_data, ratio, padding = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        output = outputs[0][0].T
        scores_all = output[:, 4:]
        max_scores = np.max(scores_all, axis=1)
        mask = max_scores > self.conf_threshold
        filtered_output, filtered_scores = output[mask], max_scores[mask]
        filtered_class_ids = np.argmax(scores_all[mask], axis=1)
        if len(filtered_output) == 0: return []
        boxes = []
        for pred in filtered_output:
            cx, cy, bw, bh = pred[:4]
            x1 = (cx - bw/2 - padding[0]) / ratio
            y1 = (cy - bh/2 - padding[1]) / ratio
            x2 = (cx + bw/2 - padding[0]) / ratio
            y2 = (cy + bh/2 - padding[1]) / ratio
            boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes)
        indices = self.nms(boxes, filtered_scores, self.iou_threshold)
        detections = []
        for i in indices:
            label = self.labels[filtered_class_ids[i]] if filtered_class_ids[i] < len(self.labels) else f"ID_{filtered_class_ids[i]}"
            det = {"label": label, "confidence": float(filtered_scores[i]), "box": (int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3]))}
            detections.append(det)
            self.log_anomaly(label, filtered_scores[i])
            self.save_anomaly_snapshot(frame, label, det["box"])
        return detections

    def save_anomaly_snapshot(self, frame, label, box):
        if not os.path.exists("onnx_detections"): os.makedirs("onnx_detections")
        filename = f"onnx_detections/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        snapshot = frame.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(snapshot, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(filename, snapshot)

class AsyncInference:
    def __init__(self, detector):
        self.detector = detector
        self.frame = None
        self.detections = []
        self.stopped = False
        self.lock = Lock()
        self.latency = 0

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.frame is not None:
                with self.lock:
                    local_frame = self.frame.copy()
                start = time.time()
                new_dets = self.detector.run_inference(local_frame)
                with self.lock:
                    self.detections = new_dets
                    self.latency = (time.time() - start) * 1000
                    self.frame = None
            else:
                time.sleep(0.01)

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_detections(self):
        with self.lock:
            return self.detections, self.latency

    def stop(self):
        self.stopped = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/best_road_anomaly.onnx")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", action="store_true", default=False, help="Save the output video to a file")
    parser.add_argument("--output", type=str, default="output_detection.mp4", help="Path to save the output video")
    parser.add_argument("--show", action="store_true", default=True, help="Display the video")
    parser.add_argument("--loop", action="store_true", default=False, help="Loop the video")
    args = parser.parse_args()

    detector = ONNXAnomalyDetector(args.model, conf_threshold=args.conf, iou_threshold=args.iou)
    async_infer = AsyncInference(detector).start()
    
    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120: video_fps = 30
    frame_delay = int(1000 / video_fps)
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    frames_processed = 0

    out = None
    if args.save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_file = args.output if args.output.endswith('.avi') else args.output.rsplit('.', 1)[0] + '.avi'
        out = cv2.VideoWriter(output_file, fourcc, video_fps, (width, height))
        if not out.isOpened():
            print("Error: Could not open VideoWriter. Trying mp4v fallback...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.output, fourcc, video_fps, (width, height))
        print(f"Saving video to: {output_file if out.isOpened() else args.output}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            async_infer.set_frame(frame)
            detections, latency = async_infer.get_detections()
            
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['label']} {det['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            model_fps = 1000 / latency if latency > 0 else 0
            
            cv2.putText(frame, f"Video FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Model FPS: {model_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if out:
                out.write(frame)
                frames_processed += 1
                if frames_processed % 30 == 0:
                    print(f"Saved {frames_processed} frames...", end="\r")
            
            if args.show:
                cv2.imshow("Async Road Anomaly Detection", frame)
                if cv2.waitKey(frame_delay) & 0xFF == ord('q'): break
    finally:
        async_infer.stop()
        cap.release()
        if out: out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
