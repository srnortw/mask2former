import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config
from models.mask2former import build_model_from_checkpoint


class Mask2FormerONNXWrapper(nn.Module):
    """
    Wraps HuggingFace Mask2Former for ONNX export.

    HF models use dict kwargs and return dataclass objects — neither is
    supported by torch.onnx.export tracing. This wrapper accepts a plain
    tensor and returns plain tensors.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.masks_queries_logits, outputs.class_queries_logits


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "checkpoints/mask2former_fp32.onnx",
    img_size: int = 512,
    opset_version: int = 13,  # opset 13 uses TorchScript exporter, compatible with static quantization
    device: str = "cpu",
) -> str:
    print(f"Loading checkpoint: {checkpoint_path}")
    model = build_model_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    wrapper = Mask2FormerONNXWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("Exporting to ONNX (this may take a few minutes)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            dynamo=False,           # force TorchScript exporter — dynamo exporter
                                    # produces detached initializers incompatible with
                                    # onnxruntime static quantization
            opset_version=opset_version,
            input_names=["pixel_values"],
            output_names=["masks_queries_logits", "class_queries_logits"],
            dynamic_axes={
                "pixel_values":         {0: "batch_size"},
                "masks_queries_logits": {0: "batch_size"},
                "class_queries_logits": {0: "batch_size"},
            },
            export_params=True,
            do_constant_folding=True,
            verbose=False,
        )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")
    return output_path


def verify_onnx(
    checkpoint_path: str,
    onnx_path: str,
    img_size: int = 512,
    tolerance: float = 1e-2,
) -> float:
    import onnx
    import onnxruntime as ort

    model_proto = onnx.load(onnx_path)
    onnx.checker.check_model(model_proto)
    print("ONNX model structure: OK")

    model = build_model_from_checkpoint(checkpoint_path)
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        pt_out = model(pixel_values=dummy)
        pt_masks  = pt_out.masks_queries_logits.numpy()
        pt_logits = pt_out.class_queries_logits.numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_masks, ort_logits = session.run(None, {"pixel_values": dummy.numpy()})

    mask_diff  = float(np.max(np.abs(pt_masks  - ort_masks)))
    logit_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    print(f"Max diff — masks: {mask_diff:.6f} | logits: {logit_diff:.6f}")

    if mask_diff < tolerance and logit_diff < tolerance:
        print("ONNX export verified.")
    else:
        print(f"WARNING: difference exceeds tolerance {tolerance}")

    return mask_diff


if __name__ == "__main__":
    cfg = load_config()
    ckpt_path = os.path.join(cfg.model.checkpoint_dir, "best_model.pth")
    fp32_path = cfg.onnx.output.fp32

    export_to_onnx(
        checkpoint_path=ckpt_path,
        output_path=fp32_path,
        img_size=cfg.onnx.input_size,
        opset_version=cfg.onnx.opset_version,
    )
    verify_onnx(ckpt_path, fp32_path, img_size=cfg.onnx.input_size)
