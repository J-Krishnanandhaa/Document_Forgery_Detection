"""
Hybrid Document Forgery Detection & Localization System

A robust hybrid (Deep Learning + Classical ML) system for multi-type 
document forgery detection and localization.

Architecture:
- Deep Learning: MobileNetV3-Small + UNet-Lite for pixel-level localization
- Classical ML: LightGBM for interpretable forgery classification
"""

__version__ = "1.0.0"

from .config import get_config
from .models import get_model, get_loss_function
from .data import get_dataset
from .features import get_feature_extractor, get_mask_refiner, get_region_extractor
from .training import get_trainer, get_metrics_tracker
from .inference import get_pipeline

__all__ = [
    'get_config',
    'get_model',
    'get_loss_function',
    'get_dataset',
    'get_feature_extractor',
    'get_mask_refiner',
    'get_region_extractor',
    'get_trainer',
    'get_metrics_tracker',
    'get_pipeline'
]
