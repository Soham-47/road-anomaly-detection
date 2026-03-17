import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_onnx_model(input_model_path, output_model_path):
    print(f"Quantizing {input_model_path}...")
    
    if not os.path.exists(input_model_path):
        print(f"Error: {input_model_path} not found.")
        return

    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"Quantization complete. Saved to: {output_model_path}")
    
    old_size = os.path.getsize(input_model_path) / (1024 * 1024)
    new_size = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"Original size: {old_size:.2f} MB")
    print(f"Quantized size: {new_size:.2f} MB")
    print(f"Size reduction: {(1 - new_size/old_size)*100:.1f}%")

if __name__ == "__main__":
    input_model = "models/best_road_anomaly.onnx"
    output_model = "models/best_road_anomaly_quantized.onnx"
    quantize_onnx_model(input_model, output_model)
