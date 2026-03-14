"""
Model Export Script

Export trained model to ONNX format for deployment.

Usage:
    python scripts/export_model.py --model outputs/checkpoints/best_doctamper.pth --format onnx
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import get_config
from src.models import get_model
from src.utils import export_to_onnx, export_to_torchscript


def parse_args():
    parser = argparse.ArgumentParser(description="Export model for deployment")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    
    parser.add_argument('--format', type=str, default='onnx',
                       choices=['onnx', 'torchscript', 'both'],
                       help='Export format')
    
    parser.add_argument('--output', type=str, default='outputs/exported',
                       help='Output directory')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = get_config(args.config)
    
    print("\n" + "="*60)
    print("Model Export")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Format: {args.format}")
    print("="*60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config).to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded")
    
    # Get image size
    image_size = config.get('data.image_size', 384)
    
    # Export
    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / 'model.onnx'
        export_to_onnx(model, str(onnx_path), input_size=(image_size, image_size))
    
    if args.format in ['torchscript', 'both']:
        ts_path = output_dir / 'model.pt'
        export_to_torchscript(model, str(ts_path), input_size=(image_size, image_size))
    
    print("\n" + "="*60)
    print("Export Complete!")
    print(f"Output: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
