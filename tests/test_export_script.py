import importlib.util
import sys
import types
from pathlib import Path

import pytest


def test_export_model_rejects_non_tflite_format():
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object
    previous = sys.modules.get("ultralytics")
    sys.modules["ultralytics"] = fake_ultralytics
    try:
        path = Path(__file__).resolve().parents[1] / "scripts" / "export_tflite.py"
        spec = importlib.util.spec_from_file_location("export_tflite", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(ValueError, match="format='tflite'"):
            module.export_model("unused.pt", format="onnx")
    finally:
        if previous is None:
            sys.modules.pop("ultralytics", None)
        else:
            sys.modules["ultralytics"] = previous
