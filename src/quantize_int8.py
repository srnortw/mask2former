import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)


class MaskCalibrationReader(CalibrationDataReader):
    """
    Feeds calibration images to ONNX Runtime to compute INT8 activation ranges.
    Required for static quantization — the model sees real data distributions,
    not just random noise. 100-200 images is sufficient; more gives no benefit.
    """

    def __init__(self, calibration_dir: str, img_size: int = 512, n_images: int = 200):
        self.img_size = img_size
        all_images = sorted([
            os.path.join(calibration_dir, f)
            for f in os.listdir(calibration_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        self.images = all_images[:n_images]
        self.idx = 0
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print(f"Calibration set: {len(self.images)} images from {calibration_dir}")

    def get_next(self):
        if self.idx >= len(self.images):
            return None

        img = cv2.imread(self.images[self.idx])
        self.idx += 1
        if img is None:
            return self.get_next()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)[np.newaxis]  # NCHW float32
        return {"pixel_values": img}


def _get_calibration_dir(cfg) -> str:
    """Use data/calibration/ if it has images, else fall back to data/raw/train/."""
    cal_dir = cfg.data.calibration_dir
    if os.path.isdir(cal_dir):
        images = [f for f in os.listdir(cal_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if images:
            return cal_dir

    raw_train = os.path.join(cfg.data.raw_dir, cfg.data.train_subdir)
    images_sub = os.path.join(raw_train, "images")
    fallback = images_sub if os.path.isdir(images_sub) else raw_train
    print(f"data/calibration/ not found — using {fallback}")
    return fallback


def quantize_int8(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    calibration_dir: str,
    img_size: int = 512,
    n_images: int = 200,
) -> str:
    from onnxruntime.quantization import shape_inference as ort_shape

    print("Starting INT8 static quantization...")
    print(f"  Input:  {fp32_onnx_path}")
    print(f"  Output: {int8_onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(int8_onnx_path)), exist_ok=True)

    # Pre-process model to embed all initializers.
    # skip_symbolic_shape=True avoids shape inference failures on Swin Transformer's
    # complex window attention patterns.
    preprocessed_path = fp32_onnx_path.replace(".onnx", "_preprocessed.onnx")
    print("  Pre-processing model for quantization...")
    ort_shape.quant_pre_process(
        fp32_onnx_path,
        preprocessed_path,
        skip_optimization=False,
        skip_symbolic_shape=True,
    )

    quantize_static(
        model_input=preprocessed_path,
        model_output=int8_onnx_path,
        calibration_data_reader=MaskCalibrationReader(calibration_dir, img_size, n_images),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    # Clean up temp file
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    fp32_mb = os.path.getsize(fp32_onnx_path) / 1e6
    int8_mb = os.path.getsize(int8_onnx_path) / 1e6
    print(f"Size: {fp32_mb:.1f} MB (fp32) → {int8_mb:.1f} MB (INT8)  ({fp32_mb/int8_mb:.1f}x smaller)")
    return int8_onnx_path


def benchmark(onnx_path: str, img_size: int = 512, n_runs: int = 30) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    for _ in range(5):  # warmup
        session.run(None, {"pixel_values": dummy})

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"pixel_values": dummy})
    ms = (time.perf_counter() - start) / n_runs * 1000

    print(f"{os.path.basename(onnx_path)}: {ms:.1f} ms/image (CPU)")
    return ms


if __name__ == "__main__":
    cfg = load_config()
    cal_dir = _get_calibration_dir(cfg)

    quantize_int8(
        fp32_onnx_path=cfg.onnx.output.fp32,
        int8_onnx_path=cfg.onnx.output.int8,
        calibration_dir=cal_dir,
        img_size=cfg.onnx.input_size,
        n_images=cfg.data.calibration.n_images,
    )

    print("\nBenchmark:")
    benchmark(cfg.onnx.output.fp32, cfg.onnx.input_size)
    benchmark(cfg.onnx.output.int8, cfg.onnx.input_size)
