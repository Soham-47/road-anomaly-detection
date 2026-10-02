import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


REPO_ROOT = Path(__file__).resolve().parents[1]


def quantize_onnx_model(input_model_path, output_model_path, overwrite=False):
    input_path = Path(input_model_path)
    output_path = Path(output_model_path)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if not input_path.exists():
        raise FileNotFoundError(f"Model not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing model: {output_path}; pass --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Quantizing {input_path} with ONNX Runtime dynamic weight quantization...")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
    )
    print(f"Saved quantized model to: {output_path}")
    old_size = input_path.stat().st_size / (1024 * 1024)
    new_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Original size: {old_size:.2f} MB")
    print(f"Quantized size: {new_size:.2f} MB")
    print(f"Size reduction: {(1 - new_size / old_size) * 100:.1f}%")
    print("Validate accuracy and latency on representative data before deployment.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Apply ONNX Runtime dynamic weight quantization")
    parser.add_argument("--input", default=str(REPO_ROOT / "models" / "best_road_anomaly.onnx"))
    parser.add_argument("--output", default=str(REPO_ROOT / "models" / "best_road_anomaly_quantized.onnx"))
    parser.add_argument("--force", action="store_true", help="Allow replacing the target file")
    args = parser.parse_args()
    quantize_onnx_model(args.input, args.output, overwrite=args.force)


if __name__ == "__main__":
    main()
