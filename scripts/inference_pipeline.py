"""
Complete Document Forgery Detection Pipeline
Implements Full Algorithm Steps 1-11

Features:
- ✅ Localization (WHERE is forgery?)
- ✅ Classification (WHAT type of forgery?)
- ✅ Confidence filtering
- ✅ Visualizations (heatmaps, overlays, bounding boxes)
- ✅ JSON output with detailed results
- ✅ Actual vs Predicted comparison (if ground truth available)

Usage:
    python scripts/inference_pipeline.py --image path/to/document.jpg
    python scripts/inference_pipeline.py --image path/to/document.jpg --ground_truth path/to/mask.png
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
import torch
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.models import get_model
from src.features import get_feature_extractor, get_mask_refiner, get_region_extractor
from src.training.classifier import ForgeryClassifier
from src.data.preprocessing import DocumentPreprocessor

# Class mapping
CLASS_NAMES = {
    0: 'Copy-Move',
    1: 'Splicing',
    2: 'Generation'
}

CLASS_COLORS = {
    0: (255, 0, 0),      # Red for Copy-Move
    1: (0, 255, 0),      # Green for Splicing
    2: (0, 0, 255)       # Blue for Generation
}


class ForgeryDetectionPipeline:
    """
    Complete forgery detection pipeline
    Implements Algorithm Steps 1-11
    """
    
    def __init__(self, config_path='config.yaml'):
        """Initialize pipeline with models"""
        print("="*70)
        print("Initializing Forgery Detection Pipeline")
        print("="*70)
        
        self.config = get_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load localization model (Steps 1-6)
        print("\n1. Loading localization model...")
        self.localization_model = get_model(self.config).to(self.device)
        checkpoint = torch.load('outputs/checkpoints/best_doctamper.pth', 
                               map_location=self.device)
        self.localization_model.load_state_dict(checkpoint['model_state_dict'])
        self.localization_model.eval()
        print(f"   ✓ Loaded (Val Dice: {checkpoint.get('best_metric', 0):.2%})")
        
        # Load classifier (Step 8)
        print("\n2. Loading forgery type classifier...")
        self.classifier = ForgeryClassifier(self.config)
        self.classifier.load('outputs/classifier')
        print("   ✓ Loaded")
        
        # Initialize components
        print("\n3. Initializing components...")
        self.preprocessor = DocumentPreprocessor(self.config, 'doctamper')
        
        # Initialize augmentation for inference
        from src.data.augmentation import DatasetAwareAugmentation
        self.augmentation = DatasetAwareAugmentation(self.config, 'doctamper', is_training=False)
        
        self.feature_extractor = get_feature_extractor(self.config, is_text_document=True)
        self.mask_refiner = get_mask_refiner(self.config)
        self.region_extractor = get_region_extractor(self.config)
        print("   ✓ Ready")
        
        print("\n" + "="*70)
        print("Pipeline Initialized Successfully!")
        print("="*70 + "\n")
    
    def detect(self, image_path, ground_truth_path=None, output_dir='outputs/inference'):
        """
        Run complete detection pipeline
        
        Args:
            image_path: Path to input document image
            ground_truth_path: Optional path to ground truth mask
            output_dir: Directory to save outputs
        
        Returns:
            results: Dictionary with detection results
        """
        print(f"\n{'='*70}")
        print(f"Processing: {image_path}")
        print(f"{'='*70}\n")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get base filename
        base_name = Path(image_path).stem
        
        # Step 1-2: Load and preprocess image (EXACTLY like dataset)
        print("Step 1-2: Loading and preprocessing...")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create dummy mask for preprocessing
        dummy_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        
        # Step 1: Preprocess (like dataset line: image, mask = self.preprocessor(image, mask))
        preprocessed_img, preprocessed_mask = self.preprocessor(image_rgb, dummy_mask)
        
        # Step 2: Augment (like dataset line: augmented = self.augmentation(image, mask))
        augmented = self.augmentation(preprocessed_img, preprocessed_mask)
        
        # Step 3: Extract tensor (like dataset line: image = augmented['image'])
        image_tensor = augmented['image']
        
        print(f"   ✓ Image shape: {image_rgb.shape}")
        print(f"   ✓ Preprocessed tensor shape: {image_tensor.shape}")
        print(f"   ✓ Tensor range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
        
        # Load ground truth if provided
        ground_truth = None
        if ground_truth_path:
            ground_truth = cv2.imread(str(ground_truth_path), cv2.IMREAD_GRAYSCALE)
            if ground_truth is not None:
                # Resize to match preprocessed size
                target_size = (image_tensor.shape[2], image_tensor.shape[1])  # (W, H)
                ground_truth = cv2.resize(ground_truth, target_size)
                print(f"   ✓ Ground truth loaded")
        
        # Step 3-4: Localization (WHERE is forgery?)
        print("\nStep 3-4: Forgery localization...")
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, decoder_features = self.localization_model(image_batch)
            prob_map = torch.sigmoid(logits).cpu().numpy()[0, 0]
        
        print(f"   ✓ Probability map generated")
        print(f"   ✓ Prob map range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
        
        # Step 5: Binary mask generation
        print("\nStep 5: Generating binary mask...")
        binary_mask = (prob_map > 0.5).astype(np.uint8)
        refined_mask = self.mask_refiner.refine(binary_mask)
        print(f"   ✓ Mask refined")
        
        # Step 6: Region extraction
        print("\nStep 6: Extracting forgery regions...")
        # Convert tensor to numpy for region extraction and feature extraction
        preprocessed_numpy = image_tensor.permute(1, 2, 0).cpu().numpy()
        regions = self.region_extractor.extract(refined_mask, prob_map, preprocessed_numpy)
        print(f"   ✓ Found {len(regions)} regions")
        
        if len(regions) == 0:
            print("\n⚠ No forgery regions detected!")
            # Still create visualizations if ground truth exists
            if ground_truth is not None:
                print("\nCreating comparison with ground truth...")
                self._create_comparison_visualization(
                    image_rgb, prob_map, refined_mask, ground_truth, 
                    base_name, output_path
                )
            return self._create_clean_result(image_rgb, base_name, output_path, ground_truth)
        
        # Step 7-8: Feature extraction and classification
        print("\nStep 7-8: Classifying forgery types...")
        region_results = []
        
        for i, region in enumerate(regions):
            # Extract features (Step 7)
            features = self.feature_extractor.extract(
                preprocessed_numpy,
                region['region_mask'],
                [f.cpu() for f in decoder_features]
            )
            
            # Ensure correct dimension (526)
            expected_dim = 526
            if len(features) < expected_dim:
                features = np.pad(features, (0, expected_dim - len(features)))
            elif len(features) > expected_dim:
                features = features[:expected_dim]
            
            features = features.reshape(1, -1)
            
            # Classify (Step 8)
            predictions, confidences = self.classifier.predict(features)
            forgery_type = int(predictions[0])
            confidence = float(confidences[0])
            
            region_results.append({
                'region_id': i + 1,
                'bounding_box': region['bounding_box'],
                'area': int(region['area']),
                'forgery_type': CLASS_NAMES[forgery_type],
                'forgery_type_id': forgery_type,
                'confidence': confidence,
                'mask_probability_mean': float(prob_map[region['region_mask'] > 0].mean())
            })
            
            print(f"   Region {i+1}: {CLASS_NAMES[forgery_type]} "
                  f"(confidence: {confidence:.2%})")
        
        # Step 9: False positive removal
        print("\nStep 9: Filtering low-confidence regions...")
        confidence_threshold = self.config.get('classification.confidence_threshold', 0.6)
        filtered_results = [r for r in region_results if r['confidence'] >= confidence_threshold]
        print(f"   ✓ Kept {len(filtered_results)}/{len(region_results)} regions "
              f"(threshold: {confidence_threshold:.0%})")
        
        # Step 10-11: Generate outputs
        print("\nStep 10-11: Generating outputs...")
        
        # Calculate scale factors for coordinate conversion
        # Bounding boxes are in preprocessed coordinates (384x384)
        # Need to scale to original image coordinates
        orig_h, orig_w = image_rgb.shape[:2]
        prep_h, prep_w = prob_map.shape
        scale_x = orig_w / prep_w
        scale_y = orig_h / prep_h
        
        # Create visualizations
        self._create_visualizations(
            image_rgb, prob_map, refined_mask, filtered_results,
            ground_truth, base_name, output_path, scale_x, scale_y
        )
        
        # Create JSON output
        results = self._create_json_output(
            image_path, filtered_results, ground_truth, base_name, output_path
        )
        
        print(f"\n{'='*70}")
        print("✅ Detection Complete!")
        print(f"{'='*70}")
        print(f"Output directory: {output_path}")
        print(f"Detected {len(filtered_results)} forgery regions")
        print(f"{'='*70}\n")
        
        return results
    
    def _create_visualizations(self, image, prob_map, mask, results, 
                               ground_truth, base_name, output_path, scale_x, scale_y):
        """Create all visualizations"""
        
        # 1. Probability heatmap
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Document')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        plt.colorbar(label='Forgery Probability')
        plt.title('Probability Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('Binary Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{base_name}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved heatmap")
        
        # 2. Overlay with bounding boxes and labels
        overlay = image.copy()
        alpha = 0.4
        
        # Create colored mask overlay (scale mask to original size)
        mask_scaled = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(image)
        
        for result in results:
            bbox = result['bounding_box']
            forgery_type = result['forgery_type_id']
            color = CLASS_COLORS[forgery_type]
            
            # Scale bounding box to original image coordinates
            x, y, w, h = bbox
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            
            # Color the region
            colored_mask[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = color
        
        # Blend with original
        overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
        
        # Draw bounding boxes and labels
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(overlay)
        
        for result in results:
            bbox = result['bounding_box']
            x, y, w, h = bbox  # bbox is [x, y, w, h] in preprocessed coordinates
            
            # Scale to original image coordinates
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            w_scaled = w * scale_x
            h_scaled = h * scale_y
            
            forgery_type = result['forgery_type']
            confidence = result['confidence']
            color_rgb = tuple(c/255 for c in CLASS_COLORS[result['forgery_type_id']])
            
            # Draw rectangle
            rect = patches.Rectangle((x_scaled, y_scaled), w_scaled, h_scaled,
                                     linewidth=2, edgecolor=color_rgb,
                                     facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            label = f"{forgery_type}\n{confidence:.1%}"
            ax.text(x_scaled, y_scaled-10, label, color='white', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor=color_rgb, alpha=0.8))
        
        ax.axis('off')
        ax.set_title('Forgery Detection Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / f'{base_name}_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved overlay")
        
        # 3. Comparison with ground truth (if available)
        if ground_truth is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(image)
            axes[0].set_title('Original Document', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(ground_truth, cmap='gray')
            axes[1].set_title('Ground Truth', fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title('Predicted Mask', fontsize=12)
            axes[2].axis('off')
            
            # Calculate metrics
            intersection = np.logical_and(ground_truth > 127, mask > 0).sum()
            union = np.logical_or(ground_truth > 127, mask > 0).sum()
            iou = intersection / (union + 1e-8)
            dice = 2 * intersection / (ground_truth.sum() + mask.sum() + 1e-8)
            
            fig.suptitle(f'Actual vs Predicted (IoU: {iou:.2%}, Dice: {dice:.2%})',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path / f'{base_name}_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved comparison (IoU: {iou:.2%}, Dice: {dice:.2%})")
        
        # 4. Per-region visualization
        if len(results) > 0:
            n_regions = len(results)
            cols = min(4, n_regions)
            rows = (n_regions + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if n_regions == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, result in enumerate(results):
                bbox = result['bounding_box']
                x, y, w, h = bbox  # bbox is [x, y, w, h] in preprocessed coordinates
                
                # Scale to original image coordinates
                x_scaled = int(x * scale_x)
                y_scaled = int(y * scale_y)
                w_scaled = int(w * scale_x)
                h_scaled = int(h * scale_y)
                
                region_img = image[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled]
                
                axes[i].imshow(region_img)
                axes[i].set_title(f"Region {i+1}: {result['forgery_type']}\n"
                                 f"Confidence: {result['confidence']:.1%}",
                                 fontsize=10)
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(n_regions, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path / f'{base_name}_regions.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved region details")
    
    def _create_json_output(self, image_path, results, ground_truth, base_name, output_path):
        """Create JSON output with results"""
        
        output = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'num_regions_detected': len(results),
            'regions': results
        }
        
        # Add ground truth comparison if available
        if ground_truth is not None:
            output['has_ground_truth'] = True
        
        # Save JSON
        json_path = output_path / f'{base_name}_results.json'
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"   ✓ Saved JSON results")
        
        return output
    
    def _create_comparison_visualization(self, image, prob_map, mask, ground_truth, 
                                         base_name, output_path):
        """Create comparison visualization between actual and predicted"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Document', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(ground_truth, cmap='gray')
        axes[0, 1].set_title('Ground Truth (Actual)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Predicted mask
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Predicted Mask', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Probability heatmap
        im = axes[1, 1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('Probability Heatmap', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Calculate metrics
        intersection = np.logical_and(ground_truth > 127, mask > 0).sum()
        union = np.logical_or(ground_truth > 127, mask > 0).sum()
        gt_sum = (ground_truth > 127).sum()
        pred_sum = (mask > 0).sum()
        
        iou = intersection / (union + 1e-8)
        dice = 2 * intersection / (gt_sum + pred_sum + 1e-8)
        precision = intersection / (pred_sum + 1e-8) if pred_sum > 0 else 0
        recall = intersection / (gt_sum + 1e-8) if gt_sum > 0 else 0
        
        fig.suptitle(f'Actual vs Predicted Comparison\n'
                    f'IoU: {iou:.2%} | Dice: {dice:.2%} | '
                    f'Precision: {precision:.2%} | Recall: {recall:.2%}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{base_name}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved comparison (IoU: {iou:.2%}, Dice: {dice:.2%})")
    
    def _create_clean_result(self, image, base_name, output_path, ground_truth=None):
        """Create result for clean (no forgery) document"""
        
        # Save original image
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title('No Forgery Detected', fontsize=14, fontweight='bold', color='green')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path / f'{base_name}_clean.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create JSON
        output = {
            'timestamp': datetime.now().isoformat(),
            'num_regions_detected': 0,
            'regions': [],
            'status': 'clean'
        }
        
        json_path = output_path / f'{base_name}_results.json'
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return output


def main():
    parser = argparse.ArgumentParser(description='Document Forgery Detection Pipeline')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input document image')
    parser.add_argument('--ground_truth', type=str, default=None,
                       help='Path to ground truth mask (optional)')
    parser.add_argument('--output_dir', type=str, default='outputs/inference',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ForgeryDetectionPipeline(args.config)
    
    # Run detection
    results = pipeline.detect(
        args.image,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\nDetection Summary:")
    print(f"  Regions detected: {results['num_regions_detected']}")
    if results['num_regions_detected'] > 0:
        for region in results['regions']:
            print(f"    - {region['forgery_type']}: {region['confidence']:.1%} confidence")


if __name__ == '__main__':
    main()
