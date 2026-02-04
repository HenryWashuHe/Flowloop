"""
ONNX Export Script for FlowLoop Models.

Exports trained PyTorch models to ONNX format for inference.

Usage:
    python export_onnx.py --model emotion --checkpoint ./outputs/best_model.pt --output ./models

Features:
- Export EmotionNet and FusionNet to ONNX
- Model verification with ONNX checker
- Test inference with ONNX Runtime
- Optional INT8 quantization
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.emotion_net import EmotionNet
from models.fusion_net import FeatureFusionNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_emotion_net(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 13,
) -> Path:
    """
    Export EmotionNet to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output directory
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load model
    model = EmotionNet(pretrained=False)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded trained weights")
    else:
        logger.warning("Checkpoint not found, exporting with random weights")

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export path
    onnx_path = output_path / "emotion_net.onnx"

    # Export
    logger.info("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["image"],
        output_names=["emotions", "frustration", "engagement", "features"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "emotions": {0: "batch_size"},
            "frustration": {0: "batch_size"},
            "engagement": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Verify
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")

    # Test inference
    logger.info("Testing inference with ONNX Runtime...")
    _test_onnx_inference(onnx_path, dummy_input.numpy())

    logger.info(f"EmotionNet exported to {onnx_path}")
    return onnx_path


def export_fusion_net(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 13,
) -> Path:
    """
    Export FusionNet to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output directory
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model
    """
    logger.info(f"Loading fusion model from {checkpoint_path}")

    # Load model
    model = FeatureFusionNet()

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded trained weights")
    else:
        logger.warning("Checkpoint not found, exporting with random weights")

    model.eval()

    # Create dummy inputs
    dummy_geometric = torch.randn(1, 20)
    dummy_emotion = torch.randn(1, 1280)

    # Export path
    onnx_path = output_path / "fusion_net.onnx"

    # Custom forward for export (tuple input)
    class FusionWrapper(torch.nn.Module):
        def __init__(self, model: FeatureFusionNet):
            super().__init__()
            self.model = model

        def forward(
            self, geometric: torch.Tensor, emotion: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = self.model(geometric, emotion)
            return outputs["attention"], outputs["frustration"]

    wrapper = FusionWrapper(model)

    # Export
    logger.info("Exporting to ONNX...")
    torch.onnx.export(
        wrapper,
        (dummy_geometric, dummy_emotion),
        str(onnx_path),
        input_names=["geometric_features", "emotion_features"],
        output_names=["attention", "frustration"],
        dynamic_axes={
            "geometric_features": {0: "batch_size"},
            "emotion_features": {0: "batch_size"},
            "attention": {0: "batch_size"},
            "frustration": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Verify
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")

    logger.info(f"FusionNet exported to {onnx_path}")
    return onnx_path


def _test_onnx_inference(onnx_path: Path, test_input: np.ndarray) -> None:
    """Test ONNX model inference."""
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: test_input.astype(np.float32)})

    logger.info(f"  Input shape: {test_input.shape}")
    for i, output in enumerate(outputs):
        logger.info(f"  Output {i} shape: {output.shape}")


def quantize_model(
    onnx_path: Path,
    output_path: Path,
    quantization_type: str = "dynamic",
) -> Path:
    """
    Quantize ONNX model to INT8.

    Args:
        onnx_path: Path to ONNX model
        output_path: Output path for quantized model
        quantization_type: "dynamic" or "static"

    Returns:
        Path to quantized model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_path = output_path.parent / f"{output_path.stem}_int8.onnx"

    logger.info(f"Quantizing {onnx_path} with {quantization_type} quantization...")

    if quantization_type == "dynamic":
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )
    else:
        raise NotImplementedError("Static quantization requires calibration data")

    # Compare sizes
    original_size = onnx_path.stat().st_size / (1024 * 1024)
    quantized_size = quantized_path.stat().st_size / (1024 * 1024)

    logger.info(f"Original size: {original_size:.2f} MB")
    logger.info(f"Quantized size: {quantized_size:.2f} MB")
    logger.info(f"Compression ratio: {original_size / quantized_size:.2f}x")

    return quantized_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        choices=["emotion", "fusion", "all"],
        default="all",
        help="Which model to export",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs/best_model.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models",
        help="Output directory",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)

    if args.model in ["emotion", "all"]:
        emotion_path = export_emotion_net(checkpoint_path, output_dir)
        if args.quantize:
            quantize_model(emotion_path, emotion_path)

    if args.model in ["fusion", "all"]:
        fusion_checkpoint = checkpoint_path.parent / "fusion_model.pt"
        fusion_path = export_fusion_net(fusion_checkpoint, output_dir)
        if args.quantize:
            quantize_model(fusion_path, fusion_path)

    logger.info("\nExport complete!")


if __name__ == "__main__":
    main()
