from ultralytics import YOLO
import shutil
import os

def export_model(model_path, format="tflite", imgsz=320, int8=False):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Exporting to {format} with imgsz={imgsz}, int8={int8}...")
    
    path = model.export(
        format=format, 
        imgsz=imgsz, 
        int8=int8,
        data="data/yolo_format/data.yaml"
    )
    
    suffix = "_int8" if int8 else "_float32"
    target_path = f"models/best_road_anomaly{suffix}.tflite"
    
    if os.path.exists(path):
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".tflite"):
                    shutil.move(os.path.join(path, f), target_path)
                    print(f"Moved exported model to: {target_path}")
                    break
        else:
            shutil.move(path, target_path)
            print(f"Moved exported model to: {target_path}")
    else:
        print(f"Export path not found: {path}")

if __name__ == "__main__":
    model_path = "models/best_road_anomaly.pt"
    export_model(model_path, int8=True)
