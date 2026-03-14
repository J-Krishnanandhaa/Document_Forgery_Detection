"""
Dataset-aware loss functions
Implements Critical Fix #2: Dataset-Aware Loss Function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice loss
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss
        
        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
        
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss for segmentation
    Dataset-aware: Only uses Dice when pixel masks are available
    """
    
    def __init__(self, 
                 bce_weight: float = 1.0,
                 dice_weight: float = 1.0):
        """
        Initialize combined loss
        
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
        """
        super().__init__()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                has_pixel_mask: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute loss (dataset-aware)
        
        Critical Fix #2: Only use Dice loss for datasets with pixel masks
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
            has_pixel_mask: Whether dataset has pixel-level masks
        
        Returns:
            Dictionary with 'total', 'bce', and optionally 'dice' losses
        """
        # BCE loss (always used)
        bce = self.bce_loss(pred, target)
        
        losses = {
            'bce': bce
        }
        
        if has_pixel_mask:
            # Use Dice loss only for datasets with pixel masks
            dice = self.dice_loss(pred, target)
            losses['dice'] = dice
            losses['total'] = self.bce_weight * bce + self.dice_weight * dice
        else:
            # Critical Fix #2: CASIA only uses BCE
            losses['total'] = self.bce_weight * bce
        
        return losses


class DatasetAwareLoss(nn.Module):
    """
    Dataset-aware loss function wrapper
    Automatically determines appropriate loss based on dataset metadata
    """
    
    def __init__(self, config):
        """
        Initialize dataset-aware loss
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        
        bce_weight = config.get('loss.bce_weight', 1.0)
        dice_weight = config.get('loss.dice_weight', 1.0)
        
        self.combined_loss = CombinedLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight
        )
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                metadata: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute loss with dataset awareness
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
            metadata: Batch metadata containing 'has_pixel_mask' flags
        
        Returns:
            Dictionary with loss components
        """
        # Check if batch has pixel masks
        has_pixel_mask = all(m.get('has_pixel_mask', True) for m in metadata) \
                        if isinstance(metadata, list) else metadata.get('has_pixel_mask', True)
        
        return self.combined_loss(pred, target, has_pixel_mask)


def get_loss_function(config) -> DatasetAwareLoss:
    """
    Factory function to create loss
    
    Args:
        config: Configuration object
    
    Returns:
        Loss function instance
    """
    return DatasetAwareLoss(config)
