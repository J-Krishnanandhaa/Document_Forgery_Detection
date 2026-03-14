"""
Dataset-aware augmentation for training
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional


class DatasetAwareAugmentation:
    """Dataset-aware augmentation pipeline"""
    
    def __init__(self, config, dataset_name: str, is_training: bool = True):
        """
        Initialize augmentation pipeline
        
        Args:
            config: Configuration object
            dataset_name: Dataset name
            is_training: Whether in training mode
        """
        self.config = config
        self.dataset_name = dataset_name
        self.is_training = is_training
        
        # Build augmentation pipeline
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        """Build albumentations transform pipeline"""
        
        transforms = []
        
        if self.is_training and self.config.get('augmentation.enabled', True):
            # Common augmentations
            common_augs = self.config.get('augmentation.common', [])
            
            for aug_config in common_augs:
                aug_type = aug_config.get('type')
                prob = aug_config.get('prob', 0.5)
                
                if aug_type == 'noise':
                    transforms.append(
                        A.GaussNoise(var_limit=(10.0, 50.0), p=prob)
                    )
                
                elif aug_type == 'motion_blur':
                    transforms.append(
                        A.MotionBlur(blur_limit=7, p=prob)
                    )
                
                elif aug_type == 'jpeg_compression':
                    quality_range = aug_config.get('quality', [60, 95])
                    transforms.append(
                        A.ImageCompression(quality_lower=quality_range[0],
                                          quality_upper=quality_range[1],
                                          p=prob)
                    )
                
                elif aug_type == 'lighting':
                    transforms.append(
                        A.OneOf([
                            A.RandomBrightnessContrast(p=1.0),
                            A.RandomGamma(p=1.0),
                            A.HueSaturationValue(p=1.0),
                        ], p=prob)
                    )
                
                elif aug_type == 'perspective':
                    transforms.append(
                        A.Perspective(scale=(0.02, 0.05), p=prob)
                    )
            
            # Dataset-specific augmentations
            if self.dataset_name == 'receipts':
                receipt_augs = self.config.get('augmentation.receipts', [])
                
                for aug_config in receipt_augs:
                    aug_type = aug_config.get('type')
                    prob = aug_config.get('prob', 0.5)
                    
                    if aug_type == 'stain':
                        # Simulate stains using random blobs
                        transforms.append(
                            A.RandomShadow(
                                shadow_roi=(0, 0, 1, 1),
                                num_shadows_lower=1,
                                num_shadows_upper=3,
                                shadow_dimension=5,
                                p=prob
                            )
                        )
                    
                    elif aug_type == 'fold':
                        # Simulate folds using grid distortion
                        transforms.append(
                            A.GridDistortion(num_steps=5, distort_limit=0.1, p=prob)
                        )
        
        # Always convert to tensor
        transforms.append(ToTensorV2())
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'}
        )
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Apply augmentation
        
        Args:
            image: Input image (H, W, 3), float32, [0, 1]
            mask: Optional mask (H, W), uint8, {0, 1}
        
        Returns:
            Dictionary with 'image' and optionally 'mask'
        """
        # Convert to uint8 for albumentations
        image_uint8 = (image * 255).astype(np.uint8)
        
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            augmented = self.transform(image=image_uint8, mask=mask_uint8)
            
            # Convert back to float32
            augmented['image'] = augmented['image'].float() / 255.0
            augmented['mask'] = (augmented['mask'].float() / 255.0).unsqueeze(0)
        else:
            augmented = self.transform(image=image_uint8)
            augmented['image'] = augmented['image'].float() / 255.0
        
        return augmented


def get_augmentation(config, dataset_name: str, is_training: bool = True) -> DatasetAwareAugmentation:
    """
    Get augmentation pipeline
    
    Args:
        config: Configuration object
        dataset_name: Dataset name
        is_training: Whether in training mode
    
    Returns:
        Augmentation pipeline
    """
    return DatasetAwareAugmentation(config, dataset_name, is_training)
