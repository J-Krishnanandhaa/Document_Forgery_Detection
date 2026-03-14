"""
UNet-Lite Decoder for forgery localization
Lightweight decoder with skip connections, depthwise separable convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    """Single decoder block with skip connection"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Initialize decoder block
        
        Args:
            in_channels: Input channels from previous decoder stage
            skip_channels: Channels from encoder skip connection
            out_channels: Output channels
        """
        super().__init__()
        
        # Combine upsampled features with skip connection
        combined_channels = in_channels + skip_channels
        
        self.conv1 = DepthwiseSeparableConv(combined_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input from previous decoder stage
            skip: Skip connection from encoder
        
        Returns:
            Decoded features
        """
        # Bilinear upsampling
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class UNetLiteDecoder(nn.Module):
    """
    UNet-Lite decoder for forgery localization
    
    Features:
    - Skip connections from encoder stages
    - Bilinear upsampling
    - Depthwise separable convolutions for efficiency
    """
    
    def __init__(self, 
                 encoder_channels: List[int],
                 decoder_channels: List[int] = None,
                 output_channels: int = 1):
        """
        Initialize decoder
        
        Args:
            encoder_channels: List of encoder feature channels [stage0, ..., stageN]
            decoder_channels: List of decoder output channels
            output_channels: Number of output channels (1 for binary mask)
        """
        super().__init__()
        
        # Default decoder channels if not provided
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]
        
        # Reverse encoder channels for decoder (bottom to top)
        encoder_channels = encoder_channels[::-1]
        
        # Initial convolution from deepest encoder features
        self.initial_conv = DepthwiseSeparableConv(encoder_channels[0], decoder_channels[0])
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(encoder_channels) - 1):
            in_ch = decoder_channels[i]
            skip_ch = encoder_channels[i + 1]
            out_ch = decoder_channels[i + 1] if i + 1 < len(decoder_channels) else decoder_channels[-1]
            
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch)
            )
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            DepthwiseSeparableConv(decoder_channels[-1], decoder_channels[-1]),
            nn.Conv2d(decoder_channels[-1], output_channels, kernel_size=1)
        )
        
        # Store decoder feature channels for feature extraction
        self.decoder_channels = decoder_channels
        
        print(f"UNet-Lite decoder initialized")
        print(f"Encoder channels: {encoder_channels[::-1]}")
        print(f"Decoder channels: {decoder_channels}")
    
    def forward(self, encoder_features: List[torch.Tensor]) -> tuple:
        """
        Forward pass
        
        Args:
            encoder_features: List of encoder features [stage0, ..., stageN]
        
        Returns:
            output: Forgery probability map (B, 1, H, W)
            decoder_features: List of decoder features for hybrid extraction
        """
        # Reverse for bottom-up decoding
        features = encoder_features[::-1]
        
        # Initial convolution
        x = self.initial_conv(features[0])
        
        # Store decoder features for hybrid feature extraction
        decoder_features = [x]
        
        # Decoder blocks with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, features[i + 1])
            decoder_features.append(x)
        
        # Final upsampling to original resolution
        # Assume input was 384x384, final feature map should match
        target_size = encoder_features[0].shape[2] * 2  # First encoder feature is at 1/2 scale
        x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        output = self.final_upsample[1](self.final_upsample[0](x))
        
        return output, decoder_features


def get_decoder(encoder_channels: List[int], config) -> UNetLiteDecoder:
    """
    Factory function to create decoder
    
    Args:
        encoder_channels: Encoder feature channels
        config: Configuration object
    
    Returns:
        Decoder instance
    """
    output_channels = config.get('model.output_channels', 1)
    return UNetLiteDecoder(encoder_channels, output_channels=output_channels)
