"""
Model export utilities (ONNX, quantization)
"""

import torch
import torch.onnx
from pathlib import Path
from typing import Tuple


def export_to_onnx(model: torch.nn.Module,
                  save_path: str,
                  input_size: Tuple[int, int] = (384, 384),
                  opset_version: int = 14):
    """
    Export model to ONNX format
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        input_size: Input image size (H, W)
        opset_version: ONNX opset version
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output', 'features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {save_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully")
    except ImportError:
        print("onnx package not installed, skipping verification")


def export_to_torchscript(model: torch.nn.Module,
                         save_path: str,
                         input_size: Tuple[int, int] = (384, 384)):
    """
    Export model to TorchScript format
    
    Args:
        model: PyTorch model
        save_path: Path to save model
        input_size: Input image size (H, W)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Script the model
    scripted_model = torch.jit.trace(model, dummy_input)
    
    # Save
    scripted_model.save(save_path)
    
    print(f"Model exported to TorchScript: {save_path}")


def quantize_model(model: torch.nn.Module,
                  save_path: str,
                  quantization_type: str = 'dynamic'):
    """
    Quantize model for faster inference
    
    Args:
        model: PyTorch model
        save_path: Path to save quantized model
        quantization_type: 'dynamic' or 'static'
    """
    model.eval()
    model = model.cpu()
    
    if quantization_type == 'dynamic':
        # Dynamic quantization (easiest)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")
    
    # Save
    torch.save(quantized_model.state_dict(), save_path)
    
    print(f"Quantized model saved to: {save_path}")
    
    return quantized_model
