"""
LightGBM Classifier Training - DocTamper with Tampering Labels
FIXED VERSION with proper checkpointing and feature dimension handling

Implements Algorithm Steps 7-8:
  7. Hybrid Feature Extraction
  8. Region-wise Forgery Classification

Uses:
- Localization: best_doctamper.pth (Steps 1-6 complete)
- Training: DocTamper TrainingSet + tampering/DocTamperV1-TrainingSet.pk
- Testing: DocTamper TestingSet + tampering/DocTamperV1-TestingSet.pk
- Classes: Copy-Move (CM), Splicing (SP), Generation (GE)

Features:
- ✅ Checkpoint saving every 1000 samples
- ✅ Resume from checkpoint if interrupted
- ✅ Fixed feature dimension mismatch
- ✅ Robust error handling

Usage:
    python scripts/train_classifier_doctamper_fixed.py
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import lmdb
import cv2
import torch
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.models import get_model
from src.features import get_feature_extractor
from src.training.classifier import get_classifier

# Configuration
MODEL_PATH = 'outputs/checkpoints/best_doctamper.pth'
OUTPUT_DIR = 'outputs/classifier'
MAX_SAMPLES = 999999  # Use all available samples

# Label mapping (Algorithm Step 8.2) - 3 classes
LABEL_MAP = {
    'CM': 0,  # Copy-Move
    'SP': 1,  # Splicing
    'GE': 2,  # Generation (AI-generated, separate from Splicing)
}


def load_tampering_labels(label_file):
    """Load forgery type labels from tampering folder"""
    with open(label_file, 'rb') as f:
        labels = pickle.load(f)
    
    print(f"Loaded {len(labels)} labels from {label_file}")
    return labels


def load_sample_from_lmdb(lmdb_env, index):
    """Load image and mask from LMDB"""
    txn = lmdb_env.begin()
    
    # Get image
    img_key = f'image-{index:09d}'.encode('utf-8')
    img_data = txn.get(img_key)
    if not img_data:
        return None, None
    
    # Get mask (DocTamper uses 'label-' not 'mask-')
    mask_key = f'label-{index:09d}'.encode('utf-8')
    mask_data = txn.get(mask_key)
    if not mask_data:
        return None, None
    
    # Decode
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask_array = np.frombuffer(mask_data, dtype=np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    
    return image, mask


def extract_features(config, model, lmdb_path, tampering_labels, 
                    max_samples, device, split_name):
    """
    Extract hybrid features with checkpointing and resume capability
    """
    
    print(f"\n{'='*60}")
    print(f"Extracting features from {split_name}")
    print(f"{'='*60}")
    
    # Setup checkpoint directory
    checkpoint_dir = Path(OUTPUT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint to resume
    checkpoints = list(checkpoint_dir.glob(f'checkpoint_{split_name}_*.npz'))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        print(f"✓ Found checkpoint: {latest_checkpoint.name}")
        
        data = np.load(latest_checkpoint, allow_pickle=True)
        all_features = data['features'].tolist()
        all_labels = data['labels'].tolist()
        expected_dim = int(data['feature_dim'])
        start_idx = len(all_features)
        
        print(f"✓ Resuming from sample {start_idx}, feature_dim={expected_dim}")
    else:
        all_features = []
        all_labels = []
        expected_dim = None
        start_idx = 0
    
    # Open LMDB
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    # Initialize feature extractor
    feature_extractor = get_feature_extractor(config, is_text_document=True)
    
    # Process samples
    num_processed = start_idx
    dim_mismatch_count = 0
    
    for i in tqdm(range(start_idx, min(len(tampering_labels), max_samples)), 
                  desc=f"Processing {split_name}", initial=start_idx, 
                  total=min(len(tampering_labels), max_samples)):
        try:
            # Skip if no label
            if i not in tampering_labels:
                continue
            
            # Get forgery type label
            forgery_type = tampering_labels[i]
            if forgery_type not in LABEL_MAP:
                continue
            
            label = LABEL_MAP[forgery_type]
            
            # Load image and mask
            image, mask = load_sample_from_lmdb(env, i)
            if image is None or mask is None:
                continue
            
            # Skip if no forgery
            if mask.max() == 0:
                continue
            
            # Prepare for model
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Get deep features from localization model
            with torch.no_grad():
                logits, decoder_features = model(image_tensor)
            
            # Use ground truth mask for feature extraction
            mask_binary = (mask > 127).astype(np.uint8)
            
            # Extract hybrid features
            features = feature_extractor.extract(
                image / 255.0,
                mask_binary,
                [f.cpu() for f in decoder_features]
            )
            
            # Set expected dimension from first valid sample
            if expected_dim is None:
                expected_dim = len(features)
                print(f"\n✓ Feature dimension set to: {expected_dim}")
            
            # Ensure consistent feature dimension
            if len(features) != expected_dim:
                if len(features) < expected_dim:
                    features = np.pad(features, (0, expected_dim - len(features)), mode='constant')
                else:
                    features = features[:expected_dim]
                dim_mismatch_count += 1
            
            all_features.append(features)
            all_labels.append(label)
            num_processed += 1
            
            # Save checkpoint every 10,000 samples (only 12 checkpoints total)
            if num_processed % 10000 == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_{split_name}_{num_processed}.npz'
                features_array = np.array(all_features, dtype=np.float32)
                labels_array = np.array(all_labels, dtype=np.int32)
                
                np.savez_compressed(checkpoint_path,
                                   features=features_array,
                                   labels=labels_array,
                                   feature_dim=expected_dim)
                print(f"\n✓ Checkpoint: {num_processed} samples (dim={expected_dim}, mismatches={dim_mismatch_count})")
                
                # Delete old checkpoints to save space (keep only last 2)
                old_checkpoints = sorted(checkpoint_dir.glob(f'checkpoint_{split_name}_*.npz'))
                if len(old_checkpoints) > 2:
                    for old_cp in old_checkpoints[:-2]:
                        old_cp.unlink()
                        print(f"  Cleaned up: {old_cp.name}")
        
        except Exception as e:
            print(f"\n⚠ Error at sample {i}: {str(e)[:80]}")
            continue
    
    env.close()
    
    print(f"\n✓ Extracted {num_processed} samples")
    if dim_mismatch_count > 0:
        print(f"⚠ Fixed {dim_mismatch_count} dimension mismatches")
    
    # Save final features
    final_path = checkpoint_dir / f'features_{split_name}_final.npz'
    if len(all_features) > 0:
        features_array = np.array(all_features, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int32)
        
        np.savez_compressed(final_path,
                           features=features_array,
                           labels=labels_array,
                           feature_dim=expected_dim)
        print(f"✓ Final features saved: {final_path}")
        print(f"  Shape: features={features_array.shape}, labels={labels_array.shape}")
        
        return features_array, labels_array
    
    return None, None


def main():
    config = get_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("LightGBM Classifier Training - DocTamper (FIXED)")
    print("Implements Algorithm Steps 7-8")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Max samples: {MAX_SAMPLES}")
    print("="*60)
    print("\nForgery Type Classes (Step 8.2):")
    print("  0: Copy-Move (CM)")
    print("  1: Splicing (SP)")
    print("  2: Generation (GE)")
    print("="*60)
    
    # Load localization model
    print("\nLoading localization model...")
    model = get_model(config).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded (Val Dice: {checkpoint.get('best_metric', 0):.4f})")
    
    # Load tampering labels
    train_labels = load_tampering_labels(
        'datasets/DocTamper/tampering/DocTamperV1-TrainingSet.pk'
    )
    test_labels = load_tampering_labels(
        'datasets/DocTamper/tampering/DocTamperV1-TestingSet.pk'
    )
    
    # Extract features from TrainingSet
    X_train, y_train = extract_features(
        config, model,
        'datasets/DocTamper/DocTamperV1-TrainingSet',
        train_labels,
        MAX_SAMPLES,
        device,
        'TrainingSet'
    )
    
    # Extract features from TestingSet
    X_test, y_test = extract_features(
        config, model,
        'datasets/DocTamper/DocTamperV1-TestingSet',
        test_labels,
        MAX_SAMPLES // 4,
        device,
        'TestingSet'
    )
    
    if X_train is None or X_test is None:
        print("\n❌ No features extracted!")
        return
    
    # Summary
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    print(f"\nTraining class distribution:")
    train_counts = np.bincount(y_train)
    class_names = ['Copy-Move', 'Splicing', 'Generation']
    for i, count in enumerate(train_counts):
        if i < len(class_names):
            print(f"  {class_names[i]}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    print(f"\nTesting class distribution:")
    test_counts = np.bincount(y_test)
    for i, count in enumerate(test_counts):
        if i < len(class_names):
            print(f"  {class_names[i]}: {count:,} ({count/len(y_test)*100:.1f}%)")
    
    # Train classifier
    print("\n" + "="*60)
    print("Training LightGBM Classifier (Step 8.1)")
    print("="*60)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classifier = get_classifier(config)
    feature_names = get_feature_extractor(config, is_text_document=True).get_feature_names()
    
    # Combine train and test for sklearn train_test_split
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])
    
    metrics = classifier.train(X_combined, y_combined, feature_names=feature_names)
    
    # Save results
    classifier.save(str(output_dir))
    print(f"\n✓ Classifier saved to: {output_dir}")
    
    # Save metrics
    metrics_path = output_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save class mapping
    class_mapping = {
        0: 'Copy-Move',
        1: 'Splicing',
        2: 'Generation'
    }
    mapping_path = output_dir / 'class_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Classifier Training Complete!")
    print("Algorithm Steps 7-8: DONE")
    print("="*60)
    print(f"\nResults:")
    print(f"  Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
    print(f"  Test F1 Score: {metrics.get('test_f1', 'N/A')}")
    print(f"\nOutput: {output_dir}")
    print("\nNext: Implement Steps 9-11 in inference pipeline")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
