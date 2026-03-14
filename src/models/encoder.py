"""
MobileNetV3-Small Encoder for forgery localization
ImageNet pretrained, feature extraction mode
"""

import torch
import torch.nn as nn
import timm
from typing import List


class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3-Small encoder for document forgery detection
    
    Chosen for:
    - Stroke-level and texture preservation
    - Robustness to compression and blur
    - Edge and CPU deployment efficiency
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize encoder
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load MobileNetV3-Small with feature extraction
        self.backbone = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)  # All feature stages
        )
        
        # Get feature channel dimensions
        # MobileNetV3-Small: [16, 16, 24, 48, 576]
        self.feature_channels = self.backbone.feature_info.channels()
        
        print(f"MobileNetV3-Small encoder initialized")
        print(f"Feature channels: {self.feature_channels}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            List of feature tensors at different scales
        """
        features = self.backbone(x)
        return features
    
    def get_feature_channels(self) -> List[int]:
        """Get feature channel dimensions for each stage"""
        return self.feature_channels


def get_encoder(config) -> MobileNetV3Encoder:
    """
    Factory function to create encoder
    
    Args:
        config: Configuration object
    
    Returns:
        Encoder instance
    """
    pretrained = config.get('model.encoder.pretrained', True)
    return MobileNetV3Encoder(pretrained=pretrained)
