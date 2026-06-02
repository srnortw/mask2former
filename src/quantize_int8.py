import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config
from onnxruntime.quantization import quantize_dynamic, QuantType


def _get_calibration_dir(cfg) -> str:
    """Kept for future static quantization use."""
    for split in [cfg.data.val_subdir, cfg.data.test_subdir, cfg.data.train_subdir]:
        split_dir = os.path.join(cfg.data.raw_dir, split)
        images_sub = os.path.join(split_dir, "images")
        candidate = images_sub if os.path.isdir(images_sub) else split_dir
        if os.path.isdir(candidate):
            imgs = [f for f in os.listdir(candidate) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if imgs:
                return candidate
    raise FileNotFoundError("No calibration images found in raw data directory")


def quantize_int8(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    **kwargs,
) -> str:
    """
    Dynamic INT8 quantization.

    Weights → QInt8 statically (weights can be negative).
    Activations → quantized at runtime (avoids QInt8 for post-Softmax/GELU
    values which are in [0,1] — unsigned territory where QInt8 wastes half
    its range on negatives, as noted in Q-ViT / FQ-ViT literature).

    TODO (future): selective static quantization with:
        weight_type=QInt8, activation_type=QUInt8
        op_types_to_quantize=['Conv', 'MatMul', 'Gemm']
        calibration on val+test set (domain-specific, not seen during training)
    """
    print("Starting INT8 dynamic quantization...")
    print(f"  Input:  {fp32_onnx_path}")
    print(f"  Output: {int8_onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(int8_onnx_path)), exist_ok=True)

    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8,
    )

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
