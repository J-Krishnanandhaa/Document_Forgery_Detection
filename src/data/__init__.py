"""Data module"""

from .preprocessing import DocumentPreprocessor, preprocess_image
from .augmentation import DatasetAwareAugmentation, get_augmentation
from .datasets import (
    DocTamperDataset,
    RTMDataset,
    CASIADataset,
    ReceiptsDataset,
    get_dataset
)

__all__ = [
    'DocumentPreprocessor',
    'preprocess_image',
    'DatasetAwareAugmentation',
    'get_augmentation',
    'DocTamperDataset',
    'RTMDataset',
    'CASIADataset',
    'ReceiptsDataset',
    'get_dataset'
]
