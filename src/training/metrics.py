"""
Training utilities and metrics
Implements Critical Fix #9: Dataset-Aware Metric Computation
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)


class SegmentationMetrics:
    """
    Segmentation metrics (IoU, Dice)
    Only computed for datasets with pixel masks (Critical Fix #9)
    """
    
    def __init__(self):
        """Initialize metrics"""
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.intersection = 0
        self.union = 0
        self.pred_sum = 0
        self.target_sum = 0
        self.total_samples = 0
    
    def update(self, 
               pred: torch.Tensor, 
               target: torch.Tensor,
               has_pixel_mask: bool = True):
        """
        Update metrics with batch
        
        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
            has_pixel_mask: Whether to compute metrics (Critical Fix #9)
        """
        if not has_pixel_mask:
            return
        
        # Binarize predictions
        pred_binary = (pred > 0.5).float()
        
        # Compute intersection and union
        intersection = (pred_binary * target).sum().item()
        union = pred_binary.sum().item() + target.sum().item() - intersection
        
        self.intersection += intersection
        self.union += union
        self.pred_sum += pred_binary.sum().item()
        self.target_sum += target.sum().item()
        self.total_samples += pred.shape[0]
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics
        
        Returns:
            Dictionary with IoU, Dice, Precision, Recall
        """
        # IoU (Jaccard)
        iou = self.intersection / (self.union + 1e-8)
        
        # Dice (F1)
        dice = (2 * self.intersection) / (self.pred_sum + self.target_sum + 1e-8)
        
        # Precision
        precision = self.intersection / (self.pred_sum + 1e-8)
        
        # Recall
        recall = self.intersection / (self.target_sum + 1e-8)
        
        return {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall
        }


class ClassificationMetrics:
    """Classification metrics for forgery type classification"""
    
    def __init__(self, num_classes: int = 3):
        """
        Initialize metrics
        
        Args:
            num_classes: Number of forgery types
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.confidences = []
    
    def update(self, 
               pred: np.ndarray, 
               target: np.ndarray,
               confidence: Optional[np.ndarray] = None):
        """
        Update metrics with predictions
        
        Args:
            pred: Predicted class indices
            target: Ground truth class indices
            confidence: Optional prediction confidences
        """
        self.predictions.extend(pred.tolist())
        self.targets.extend(target.tolist())
        if confidence is not None:
            self.confidences.extend(confidence.tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics
        
        Returns:
            Dictionary with Accuracy, F1, Precision, Recall
        """
        if len(self.predictions) == 0:
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Accuracy
        accuracy = accuracy_score(targets, preds)
        
        # F1 score (macro and weighted)
        f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(targets, preds, average='weighted', zero_division=0)
        
        # Precision and Recall
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.tolist()
        }


class MetricsTracker:
    """Track all metrics during training"""
    
    def __init__(self, config):
        """
        Initialize metrics tracker
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.num_classes = config.get('data.num_classes', 3)
        
        self.seg_metrics = SegmentationMetrics()
        self.cls_metrics = ClassificationMetrics(self.num_classes)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': []
        }
    
    def reset(self):
        """Reset metrics for new epoch"""
        self.seg_metrics.reset()
        self.cls_metrics.reset()
    
    def update_segmentation(self, 
                           pred: torch.Tensor, 
                           target: torch.Tensor,
                           dataset_name: str):
        """Update segmentation metrics (dataset-aware)"""
        has_pixel_mask = self.config.should_compute_localization_metrics(dataset_name)
        self.seg_metrics.update(pred, target, has_pixel_mask)
    
    def update_classification(self, 
                             pred: np.ndarray, 
                             target: np.ndarray,
                             confidence: Optional[np.ndarray] = None):
        """Update classification metrics"""
        self.cls_metrics.update(pred, target, confidence)
    
    def compute_all(self) -> Dict[str, float]:
        """Compute all metrics"""
        seg = self.seg_metrics.compute()
        
        # Only include classification metrics if they have data
        if len(self.cls_metrics.predictions) > 0:
            cls = self.cls_metrics.compute()
            # Prefix classification metrics to avoid collision
            cls_prefixed = {f'cls_{k}': v for k, v in cls.items()}
            return {**seg, **cls_prefixed}
        
        return seg
    
    def log_epoch(self, epoch: int, phase: str, loss: float, metrics: Dict):
        """Log metrics for epoch"""
        prefix = f'{phase}_'
        
        self.history[f'{phase}_loss'].append(loss)
        
        if 'iou' in metrics:
            self.history[f'{phase}_iou'].append(metrics['iou'])
        if 'dice' in metrics:
            self.history[f'{phase}_dice'].append(metrics['dice'])
        if 'precision' in metrics:
            self.history[f'{phase}_precision'].append(metrics['precision'])
        if 'recall' in metrics:
            self.history[f'{phase}_recall'].append(metrics['recall'])
    
    def get_history(self) -> Dict:
        """Get full training history"""
        return self.history


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'max'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum improvement required
            mode: 'min' for loss, 'max' for metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop
        
        Args:
            value: Current metric value
        
        Returns:
            True if should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop


def get_metrics_tracker(config) -> MetricsTracker:
    """Factory function for metrics tracker"""
    return MetricsTracker(config)
