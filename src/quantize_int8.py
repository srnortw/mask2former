import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_int8(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    **kwargs,   # absorb unused calibration args for backward compat
) -> str:
    """
    Dynamic INT8 quantization — weights quantized to INT8 statically,
    activations quantized at runtime. No calibration data needed.

    Recommended over static quantization for transformer architectures
    (Swin, ViT, BERT etc.) because static quantization's shape inference
    fails on window attention's complex dynamic shapes.

    Size reduction: ~3-4x  |  CPU speedup: ~2x
    """
    print("Starting INT8 dynamic quantization...")
    print(f"  Input:  {fp32_onnx_path}")
    print(f"  Output: {int8_onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(int8_onnx_path)), exist_ok=True)

    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QUInt8,
    )

    fp32_mb = os.path.getsize(fp32_onnx_path) / 1e6
    int8_mb = os.path.getsize(int8_onnx_path) / 1e6
    print(f"Size: {fp32_mb:.1f} MB (fp32) → {int8_mb:.1f} MB (INT8)  ({fp32_mb/int8_mb:.1f}x smaller)")
    return int8_onnx_path


def _get_calibration_dir(cfg) -> str:
    """Kept for backward compatibility — not needed for dynamic quantization."""
    raw_train = os.path.join(cfg.data.raw_dir, cfg.data.train_subdir)
    images_sub = os.path.join(raw_train, "images")
    return images_sub if os.path.isdir(images_sub) else raw_train


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

    quantize_int8(
        fp32_onnx_path=cfg.onnx.output.fp32,
        int8_onnx_path=cfg.onnx.output.int8,
    )

    print("\nBenchmark:")
    benchmark(cfg.onnx.output.fp32, cfg.onnx.input_size)
    benchmark(cfg.onnx.output.int8, cfg.onnx.input_size)
