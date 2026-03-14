"""
Complete Forgery Localization Network
MobileNetV3-Small Encoder + UNet-Lite Decoder
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from .encoder import MobileNetV3Encoder
from .decoder import UNetLiteDecoder


class ForgeryLocalizationNetwork(nn.Module):
    """
    Complete network for forgery localization
    
    Architecture:
    - Encoder: MobileNetV3-Small (ImageNet pretrained)
    - Decoder: UNet-Lite with skip connections
    - Output: Single-channel forgery probability map
    """
    
    def __init__(self, config):
        """
        Initialize network
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        
        # Initialize encoder
        pretrained = config.get('model.encoder.pretrained', True)
        self.encoder = MobileNetV3Encoder(pretrained=pretrained)
        
        # Initialize decoder
        encoder_channels = self.encoder.get_feature_channels()
        output_channels = config.get('model.output_channels', 1)
        self.decoder = UNetLiteDecoder(
            encoder_channels=encoder_channels,
            output_channels=output_channels
        )
        
        print(f"ForgeryLocalizationNetwork initialized")
        print(f"Total parameters: {self.count_parameters():,}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            output: Forgery probability map (B, 1, H, W) - logits
            decoder_features: Decoder features for hybrid feature extraction
        """
        # Encode
        encoder_features = self.encoder(x)
        
        # Decode
        output, decoder_features = self.decoder(encoder_features)
        
        return output, decoder_features
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary mask
        
        Args:
            x: Input image tensor (B, 3, H, W)
            threshold: Probability threshold for binarization
        
        Returns:
            Binary mask (B, 1, H, W)
        """
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = torch.sigmoid(logits)
            mask = (probs > threshold).float()
        
        return mask
    
    def get_probability_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability map
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            Probability map (B, 1, H, W)
        """
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = torch.sigmoid(logits)
        
        return probs
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_decoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get decoder features for hybrid feature extraction
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            List of decoder features
        """
        with torch.no_grad():
            _, decoder_features = self.forward(x)
        
        return decoder_features


def get_model(config) -> ForgeryLocalizationNetwork:
    """
    Factory function to create model
    
    Args:
        config: Configuration object
    
    Returns:
        Model instance
    """
    return ForgeryLocalizationNetwork(config)
