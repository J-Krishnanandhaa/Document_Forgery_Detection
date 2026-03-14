"""
Comprehensive Evaluation on Entire TestingSet

Evaluates the complete pipeline on all 30,000 samples from TestingSet
Calculates all metrics: Accuracy, Precision, Recall, F1, IoU, Dice, etc.
No visualizations - metrics only for speed

Usage:
    python scripts/evaluate_full_testingset.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.models import get_model
from src.data import get_dataset
from src.features import get_mask_refiner, get_region_extractor
from src.training.classifier import ForgeryClassifier
from src.data.preprocessing import DocumentPreprocessor
from src.data.augmentation import DatasetAwareAugmentation

# Class mapping
CLASS_NAMES = {0: 'Copy-Move', 1: 'Splicing', 2: 'Generation'}


def calculate_metrics(pred_mask, gt_mask):
    """Calculate all segmentation metrics"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    
    tp = intersection
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    
    # Segmentation metrics
    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def main():
    print("="*80)
    print("COMPREHENSIVE EVALUATION ON ENTIRE TESTINGSET")
    print("="*80)
    print("Dataset: DocTamper TestingSet (30,000 samples)")
    print("Mode: Metrics only (no visualizations)")
    print("="*80)
    
    config = get_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print("\n1. Loading models...")
    
    # Localization model
    model = get_model(config).to(device)
    checkpoint = torch.load('outputs/checkpoints/best_doctamper.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ Localization model loaded (Dice: {checkpoint.get('best_metric', 0):.2%})")
    
    # Classifier
    classifier = ForgeryClassifier(config)
    classifier.load('outputs/classifier')
    print(f"   ✓ Classifier loaded")
    
    # Components
    preprocessor = DocumentPreprocessor(config, 'doctamper')
    augmentation = DatasetAwareAugmentation(config, 'doctamper', is_training=False)
    mask_refiner = get_mask_refiner(config)
    region_extractor = get_region_extractor(config)
    
    # Load dataset
    print("\n2. Loading TestingSet...")
    dataset = get_dataset(config, 'doctamper', split='val')  # val = TestingSet
    total_samples = len(dataset)
    print(f"   ✓ Loaded {total_samples} samples")
    
    # Initialize metrics storage
    all_metrics = []
    detection_stats = {
        'total': 0,
        'has_forgery': 0,
        'detected': 0,
        'missed': 0,
        'false_positives': 0,
        'true_negatives': 0
    }
    
    print("\n3. Running evaluation...")
    print("="*80)
    
    # Process all samples
    for idx in tqdm(range(total_samples), desc="Evaluating"):
        try:
            # Get sample from dataset
            image_tensor, mask_tensor, metadata = dataset[idx]
            
            # Ground truth
            gt_mask = mask_tensor.numpy()[0]
            gt_mask_binary = (gt_mask > 0.5).astype(np.uint8)
            has_forgery = gt_mask_binary.sum() > 0
            
            # Run localization
            with torch.no_grad():
                image_batch = image_tensor.unsqueeze(0).to(device)
                logits, _ = model(image_batch)
                prob_map = torch.sigmoid(logits).cpu().numpy()[0, 0]
            
            # Generate mask
            binary_mask = (prob_map > 0.5).astype(np.uint8)
            refined_mask = mask_refiner.refine(binary_mask)
            
            # Calculate metrics
            metrics = calculate_metrics(refined_mask, gt_mask_binary)
            metrics['sample_idx'] = idx
            metrics['has_forgery'] = has_forgery
            metrics['prob_max'] = float(prob_map.max())
            
            # Detection statistics
            detected = refined_mask.sum() > 0
            
            detection_stats['total'] += 1
            if has_forgery:
                detection_stats['has_forgery'] += 1
                if detected:
                    detection_stats['detected'] += 1
                else:
                    detection_stats['missed'] += 1
            else:
                if detected:
                    detection_stats['false_positives'] += 1
                else:
                    detection_stats['true_negatives'] += 1
            
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"\nError at sample {idx}: {str(e)[:100]}")
            continue
    
    # Calculate overall statistics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Detection statistics
    print("\n📊 DETECTION STATISTICS:")
    print("-"*80)
    print(f"Total samples: {detection_stats['total']}")
    print(f"Samples with forgery: {detection_stats['has_forgery']}")
    print(f"Samples without forgery: {detection_stats['total'] - detection_stats['has_forgery']}")
    print()
    print(f"✅ Correctly detected: {detection_stats['detected']}")
    print(f"❌ Missed detections: {detection_stats['missed']}")
    print(f"⚠️  False positives: {detection_stats['false_positives']}")
    print(f"✓  True negatives: {detection_stats['true_negatives']}")
    print()
    
    # Detection rates
    if detection_stats['has_forgery'] > 0:
        detection_rate = detection_stats['detected'] / detection_stats['has_forgery']
        miss_rate = detection_stats['missed'] / detection_stats['has_forgery']
        print(f"Detection Rate (Recall): {detection_rate:.2%} ⬆️ Higher is better")
        print(f"Miss Rate: {miss_rate:.2%} ⬇️ Lower is better")
    
    if detection_stats['detected'] + detection_stats['false_positives'] > 0:
        precision = detection_stats['detected'] / (detection_stats['detected'] + detection_stats['false_positives'])
        print(f"Precision: {precision:.2%} ⬆️ Higher is better")
    
    overall_accuracy = (detection_stats['detected'] + detection_stats['true_negatives']) / detection_stats['total']
    print(f"Overall Accuracy: {overall_accuracy:.2%} ⬆️ Higher is better")
    
    # Segmentation metrics (only for samples with forgery)
    forgery_metrics = [m for m in all_metrics if m['has_forgery']]
    
    if forgery_metrics:
        print("\n📈 SEGMENTATION METRICS (on samples with forgery):")
        print("-"*80)
        
        avg_iou = np.mean([m['iou'] for m in forgery_metrics])
        avg_dice = np.mean([m['dice'] for m in forgery_metrics])
        avg_precision = np.mean([m['precision'] for m in forgery_metrics])
        avg_recall = np.mean([m['recall'] for m in forgery_metrics])
        avg_f1 = np.mean([m['f1'] for m in forgery_metrics])
        avg_accuracy = np.mean([m['accuracy'] for m in forgery_metrics])
        
        print(f"IoU (Intersection over Union): {avg_iou:.4f} ⬆️ Higher is better (0-1)")
        print(f"Dice Coefficient: {avg_dice:.4f} ⬆️ Higher is better (0-1)")
        print(f"Pixel Precision: {avg_precision:.4f} ⬆️ Higher is better (0-1)")
        print(f"Pixel Recall: {avg_recall:.4f} ⬆️ Higher is better (0-1)")
        print(f"Pixel F1-Score: {avg_f1:.4f} ⬆️ Higher is better (0-1)")
        print(f"Pixel Accuracy: {avg_accuracy:.4f} ⬆️ Higher is better (0-1)")
        
        # Probability statistics
        avg_prob = np.mean([m['prob_max'] for m in forgery_metrics])
        print(f"\nAverage Max Probability: {avg_prob:.4f}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_dir = Path('outputs/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': detection_stats['total'],
        'detection_statistics': detection_stats,
        'detection_rate': detection_stats['detected'] / detection_stats['has_forgery'] if detection_stats['has_forgery'] > 0 else 0,
        'precision': detection_stats['detected'] / (detection_stats['detected'] + detection_stats['false_positives']) if (detection_stats['detected'] + detection_stats['false_positives']) > 0 else 0,
        'overall_accuracy': overall_accuracy,
        'segmentation_metrics': {
            'iou': float(avg_iou) if forgery_metrics else 0,
            'dice': float(avg_dice) if forgery_metrics else 0,
            'precision': float(avg_precision) if forgery_metrics else 0,
            'recall': float(avg_recall) if forgery_metrics else 0,
            'f1': float(avg_f1) if forgery_metrics else 0,
            'accuracy': float(avg_accuracy) if forgery_metrics else 0
        }
    }
    
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_path}")
    
    # Detailed metrics (optional - can be large)
    # detailed_path = output_dir / 'detailed_metrics.json'
    # with open(detailed_path, 'w') as f:
    #     json.dump(all_metrics, f, indent=2)
    # print(f"✓ Detailed metrics saved to: {detailed_path}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nKey Metrics Summary:")
    print(f"  Detection Rate: {detection_stats['detected'] / detection_stats['has_forgery']:.2%}")
    print(f"  Overall Accuracy: {overall_accuracy:.2%}")
    print(f"  Dice Score: {avg_dice:.4f}" if forgery_metrics else "  Dice Score: N/A")
    print(f"  IoU: {avg_iou:.4f}" if forgery_metrics else "  IoU: N/A")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
