"""
Model Evaluation Script

Evaluate trained model on validation/test sets with comprehensive metrics.

Usage:
    python scripts/evaluate.py --model outputs/checkpoints/best_doctamper.pth --dataset doctamper
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.models import get_model
from src.data import get_dataset
from src.training.metrics import SegmentationMetrics
from src.utils import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate forgery detection model")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['doctamper', 'rtm', 'casia', 'receipts'],
                       help='Dataset to evaluate on')
    
    parser.add_argument('--split', type=str, default='val',
                       help='Data split (val/test)')
    
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                       help='Output directory')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = get_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print("="*60)
    
    # Load model
    model = get_model(config).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded")
    
    # Load dataset
    dataset = get_dataset(config, args.dataset, split=args.split)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    metrics = SegmentationMetrics()
    has_pixel_mask = config.has_pixel_mask(args.dataset)
    
    print(f"\nEvaluating...")
    
    all_ious = []
    all_dices = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            try:
                image, mask, metadata = dataset[i]
                
                # Move to device
                image = image.unsqueeze(0).to(device)
                mask = mask.unsqueeze(0).to(device)
                
                # Forward pass
                logits, _ = model(image)
                probs = torch.sigmoid(logits)
                
                # Update metrics
                if has_pixel_mask:
                    metrics.update(probs, mask, has_pixel_mask=True)
                    
                    # Per-sample metrics
                    pred_binary = (probs > 0.5).float()
                    intersection = (pred_binary * mask).sum().item()
                    union = pred_binary.sum().item() + mask.sum().item() - intersection
                    
                    iou = intersection / (union + 1e-8)
                    dice = (2 * intersection) / (pred_binary.sum().item() + mask.sum().item() + 1e-8)
                    
                    all_ious.append(iou)
                    all_dices.append(dice)
            
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
    
    # Compute final metrics
    results = metrics.compute()
    
    # Add per-sample statistics
    if has_pixel_mask and all_ious:
        results['iou_mean'] = np.mean(all_ious)
        results['iou_std'] = np.std(all_ious)
        results['dice_mean'] = np.mean(all_dices)
        results['dice_std'] = np.std(all_dices)
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Save results
    results_path = output_dir / f'{args.dataset}_{args.split}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("="*60)


if __name__ == '__main__':
    main()
