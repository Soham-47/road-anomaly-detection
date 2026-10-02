import argparse
import re
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "yolo_format"


def _export_data_yaml():
    """Create a temporary config whose dataset root is absolute and portable."""
    template = DATA_ROOT / "data.yaml"
    text = template.read_text(encoding="utf-8")
    absolute_path = f"path: '{DATA_ROOT.as_posix()}'"
    text, replacements = re.subn(r"(?m)^path:\s*.*$", absolute_path, text, count=1)
    if replacements != 1:
        raise ValueError(f"Dataset config must contain one path entry: {template}")
    temporary = tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False)
    temporary.write(text)
    temporary.close()
    return Path(temporary.name)


def _require_calibration_images():
    missing = [
        path for path in (DATA_ROOT / "images" / "train", DATA_ROOT / "images" / "val") if not path.is_dir()
    ]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "INT8 export requires calibration images; missing "
            f"{missing_paths}. Add the dataset under {DATA_ROOT}."
        )


def export_model(model_path, format="tflite", imgsz=320, int8=False, output_dir=None, overwrite=False):
    if format != "tflite":
        raise ValueError("export_model only supports format='tflite'")
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    output_dir = Path(output_dir) if output_dir else REPO_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"best_road_anomaly_{'int8' if int8 else 'float32'}.tflite"
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing export: {target_path}; pass --force")

    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))
    if int8:
        _require_calibration_images()
    data_path = _export_data_yaml()
    print(f"Exporting to {format} with imgsz={imgsz}, int8={int8}...")
    try:
        exported = Path(
            model.export(format=format, imgsz=imgsz, int8=int8, data=str(data_path))
        )
    finally:
        data_path.unlink(missing_ok=True)

    candidates = [exported] if exported.is_file() else sorted(exported.rglob("*.tflite"))
    if not candidates:
        raise FileNotFoundError(f"No .tflite file found under export path: {exported}")
    source = candidates[0]
    if source.resolve() != target_path.resolve():
        shutil.move(str(source), str(target_path))
    print(f"Saved exported model to: {target_path}")
    return target_path


def main():
    parser = argparse.ArgumentParser(description="Export a YOLO model to TFLite")
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "best_road_anomaly.pt"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--float32", action="store_true", help="Export float32 instead of int8")
    parser.add_argument("--force", action="store_true", help="Allow replacing the target file")
    args = parser.parse_args()
    export_model(args.model, imgsz=args.imgsz, int8=not args.float32, overwrite=args.force)


if __name__ == "__main__":
    main()
