"""
Dataset loaders for document forgery detection
Implements Critical Fix #7: Image-level train/test splits
"""

import os
import lmdb
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List
import json
from PIL import Image

from .preprocessing import DocumentPreprocessor
from .augmentation import DatasetAwareAugmentation


class DocTamperDataset(Dataset):
    """
    DocTamper dataset loader (LMDB-based)
    Implements chunked loading for RAM constraints
    Uses lazy LMDB initialization for multiprocessing compatibility
    """
    
    def __init__(self, 
                 config,
                 split: str = 'train',
                 chunk_start: float = 0.0,
                 chunk_end: float = 1.0):
        """
        Initialize DocTamper dataset
        
        Args:
            config: Configuration object
            split: 'train' or 'val'
            chunk_start: Start ratio for chunked training (0.0 to 1.0)
            chunk_end: End ratio for chunked training (0.0 to 1.0)
        """
        self.config = config
        self.split = split
        self.dataset_name = 'doctamper'
        
        # Get dataset path
        dataset_config = config.get_dataset_config(self.dataset_name)
        self.data_path = Path(dataset_config['path'])
        
        # Map split to actual folder names
        if split == 'train':
            lmdb_folder = 'DocTamperV1-TrainingSet'
        elif split == 'val' or split == 'test':
            lmdb_folder = 'DocTamperV1-TestingSet'
        else:
            lmdb_folder = 'DocTamperV1-TrainingSet'
        
        self.lmdb_path = str(self.data_path / lmdb_folder)
        
        if not Path(self.lmdb_path).exists():
            raise FileNotFoundError(f"LMDB folder not found: {self.lmdb_path}")
        
        # LAZY INITIALIZATION: Don't open LMDB here (pickle issue with multiprocessing)
        # Just get the count by temporarily opening
        temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with temp_env.begin() as txn:
            stat = txn.stat()
            self.length = stat['entries'] // 2
        temp_env.close()
        
        # LMDB env will be opened lazily in __getitem__
        self._env = None
        
        # Critical Fix #7: Image-level chunking (not region-level)
        self.chunk_start = int(self.length * chunk_start)
        self.chunk_end = int(self.length * chunk_end)
        self.chunk_length = self.chunk_end - self.chunk_start
        
        print(f"DocTamper {split}: Total={self.length}, "
              f"Chunk=[{self.chunk_start}:{self.chunk_end}], "
              f"Length={self.chunk_length}")
        
        # Initialize preprocessor and augmentation
        self.preprocessor = DocumentPreprocessor(config, self.dataset_name)
        self.augmentation = DatasetAwareAugmentation(
            config, 
            self.dataset_name, 
            is_training=(split == 'train')
        )
    
    @property
    def env(self):
        """Lazy LMDB environment initialization for multiprocessing compatibility"""
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                  max_readers=32, readahead=False)
        return self._env
    
    def __len__(self) -> int:
        return self.chunk_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Get item from dataset
        
        Args:
            idx: Index within chunk
        
        Returns:
            image: (3, H, W) tensor
            mask: (1, H, W) tensor
            metadata: Dictionary with additional info
        """
        # Try to get the requested sample, skip to next if missing
        max_attempts = 10
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                # Map chunk index to global index
                global_idx = self.chunk_start + idx
                
                # Read from LMDB
                with self.env.begin() as txn:
                    # DocTamper format: image-XXXXXXXXX, label-XXXXXXXXX (9 digits, dash separator)
                    img_key = f'image-{global_idx:09d}'.encode()
                    label_key = f'label-{global_idx:09d}'.encode()
                    
                    img_buf = txn.get(img_key)
                    label_buf = txn.get(label_key)
                    
                    if img_buf is None:
                        # Sample missing, try next index
                        idx = (idx + 1) % self.chunk_length
                        continue
                    
                    # Decode image
                    img_array = np.frombuffer(img_buf, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        # Failed to decode, try next index
                        idx = (idx + 1) % self.chunk_length
                        continue
                    
                    # Decode label/mask
                    if label_buf is not None:
                        label_array = np.frombuffer(label_buf, dtype=np.uint8)
                        mask = cv2.imdecode(label_array, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            # Label might be raw bytes, create empty mask
                            mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    else:
                        # No mask found - create empty mask
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)
                
                # Successfully loaded - break out of retry loop
                break
                
            except Exception as e:
                # Something went wrong, try next index
                idx = (idx + 1) % self.chunk_length
                if attempt == max_attempts - 1:
                    # Last attempt failed, create a dummy sample
                    print(f"Warning: Could not load sample at idx {original_idx}, creating dummy sample")
                    image = np.zeros((384, 384, 3), dtype=np.float32)
                    mask = np.zeros((384, 384), dtype=np.uint8)
                    global_idx = original_idx
        
        # Preprocess
        image, mask = self.preprocessor(image, mask)
        
        # Augment
        augmented = self.augmentation(image, mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # Metadata
        metadata = {
            'dataset': self.dataset_name,
            'index': global_idx,
            'has_pixel_mask': True
        }
        
        return image, mask, metadata
    
    def __del__(self):
        """Close LMDB environment"""
        if hasattr(self, '_env') and self._env is not None:
            self._env.close()



class RTMDataset(Dataset):
    """Real Text Manipulation dataset loader"""
    
    def __init__(self, config, split: str = 'train'):
        """
        Initialize RTM dataset
        
        Args:
            config: Configuration object
            split: 'train' or 'test'
        """
        self.config = config
        self.split = split
        self.dataset_name = 'rtm'
        
        # Get dataset path
        dataset_config = config.get_dataset_config(self.dataset_name)
        self.data_path = Path(dataset_config['path'])
        
        # Load split file
        split_file = self.data_path / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        self.images_dir = self.data_path / 'JPEGImages'
        self.masks_dir = self.data_path / 'SegmentationClass'
        
        print(f"RTM {split}: {len(self.image_ids)} images")
        
        # Initialize preprocessor and augmentation
        self.preprocessor = DocumentPreprocessor(config, self.dataset_name)
        self.augmentation = DatasetAwareAugmentation(
            config,
            self.dataset_name,
            is_training=(split == 'train')
        )
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Get item from dataset"""
        image_id = self.image_ids[idx]
        
        # Load image
        img_path = self.images_dir / f'{image_id}.jpg'
        image = cv2.imread(str(img_path))
        
        # Load mask
        mask_path = self.masks_dir / f'{image_id}.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Binarize mask
        mask = (mask > 0).astype(np.uint8)
        
        # Preprocess
        image, mask = self.preprocessor(image, mask)
        
        # Augment
        augmented = self.augmentation(image, mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # Metadata
        metadata = {
            'dataset': self.dataset_name,
            'image_id': image_id,
            'has_pixel_mask': True
        }
        
        return image, mask, metadata


class CASIADataset(Dataset):
    """
    CASIA v1.0 dataset loader
    Image-level labels only (no pixel masks)
    Implements Critical Fix #6: CASIA image-level handling
    """
    
    def __init__(self, config, split: str = 'train'):
        """
        Initialize CASIA dataset
        
        Args:
            config: Configuration object
            split: 'train' or 'test'
        """
        self.config = config
        self.split = split
        self.dataset_name = 'casia'
        
        # Get dataset path
        dataset_config = config.get_dataset_config(self.dataset_name)
        self.data_path = Path(dataset_config['path'])
        
        # Load authentic and tampered images
        self.authentic_dir = self.data_path / 'Au'
        self.tampered_dir = self.data_path / 'Tp'
        
        # Get all image paths
        authentic_images = list(self.authentic_dir.glob('*.jpg')) + \
                          list(self.authentic_dir.glob('*.png'))
        tampered_images = list(self.tampered_dir.glob('*.jpg')) + \
                         list(self.tampered_dir.glob('*.png'))
        
        # Create image list with labels
        self.samples = []
        for img_path in authentic_images:
            self.samples.append((img_path, 0))  # 0 = authentic
        for img_path in tampered_images:
            self.samples.append((img_path, 1))  # 1 = tampered
        
        # Critical Fix #7: Image-level split (80/20)
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(self.samples) * 0.8)
        
        if split == 'train':
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]
        
        self.samples = [self.samples[i] for i in indices]
        
        print(f"CASIA {split}: {len(self.samples)} images")
        
        # Initialize preprocessor and augmentation
        self.preprocessor = DocumentPreprocessor(config, self.dataset_name)
        self.augmentation = DatasetAwareAugmentation(
            config,
            self.dataset_name,
            is_training=(split == 'train')
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Get item from dataset"""
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        
        # Critical Fix #6: Create image-level mask (entire image)
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * label
        
        # Preprocess
        image, mask = self.preprocessor(image, mask)
        
        # Augment
        augmented = self.augmentation(image, mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # Metadata
        metadata = {
            'dataset': self.dataset_name,
            'image_path': str(img_path),
            'has_pixel_mask': False,  # Image-level only
            'label': label
        }
        
        return image, mask, metadata


class ReceiptsDataset(Dataset):
    """Find-It-Again receipts dataset loader"""
    
    def __init__(self, config, split: str = 'train'):
        """
        Initialize receipts dataset
        
        Args:
            config: Configuration object
            split: 'train', 'val', or 'test'
        """
        self.config = config
        self.split = split
        self.dataset_name = 'receipts'
        
        # Get dataset path
        dataset_config = config.get_dataset_config(self.dataset_name)
        self.data_path = Path(dataset_config['path'])
        
        # Load split file
        split_file = self.data_path / f'{split}.json'
        with open(split_file, 'r') as f:
            self.annotations = json.load(f)
        
        print(f"Receipts {split}: {len(self.annotations)} images")
        
        # Initialize preprocessor and augmentation
        self.preprocessor = DocumentPreprocessor(config, self.dataset_name)
        self.augmentation = DatasetAwareAugmentation(
            config,
            self.dataset_name,
            is_training=(split == 'train')
        )
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Get item from dataset"""
        ann = self.annotations[idx]
        
        # Load image
        img_path = self.data_path / ann['image_path']
        image = cv2.imread(str(img_path))
        
        # Create mask from bounding boxes
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for bbox in ann.get('bboxes', []):
            x, y, w_box, h_box = bbox
            mask[y:y+h_box, x:x+w_box] = 1
        
        # Preprocess
        image, mask = self.preprocessor(image, mask)
        
        # Augment
        augmented = self.augmentation(image, mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # Metadata
        metadata = {
            'dataset': self.dataset_name,
            'image_path': str(img_path),
            'has_pixel_mask': True
        }
        
        return image, mask, metadata


class FCDDataset(DocTamperDataset):
    """FCD (Forgery Classification Dataset) loader - inherits from DocTamperDataset"""
    
    def __init__(self, config, split: str = 'train'):
        self.config = config
        self.split = split
        self.dataset_name = 'fcd'
        
        # Get dataset path from config
        dataset_config = config.get_dataset_config(self.dataset_name)
        self.data_path = Path(dataset_config['path'])
        self.lmdb_path = str(self.data_path)
        
        if not Path(self.lmdb_path).exists():
            raise FileNotFoundError(f"LMDB folder not found: {self.lmdb_path}")
        
        # Get total count
        temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with temp_env.begin() as txn:
            stat = txn.stat()
            self.length = stat['entries'] // 2  # Half are images, half are labels
        temp_env.close()
        
        self._env = None
        
        # FCD is small, no chunking needed
        self.chunk_start = 0
        self.chunk_end = self.length
        self.chunk_length = self.length
        
        print(f"FCD {split}: {self.length} samples")
        
        # Initialize preprocessor and augmentation
        self.preprocessor = DocumentPreprocessor(config, self.dataset_name)
        self.augmentation = DatasetAwareAugmentation(
            config,
            self.dataset_name,
            is_training=(split == 'train')
        )


class SCDDataset(DocTamperDataset):
    """SCD (Splicing Classification Dataset) loader - inherits from DocTamperDataset"""
    
    def __init__(self, config, split: str = 'train'):
        self.config = config
        self.split = split
        self.dataset_name = 'scd'
        
        # Get dataset path from config
        dataset_config = config.get_dataset_config(self.dataset_name)
        self.data_path = Path(dataset_config['path'])
        self.lmdb_path = str(self.data_path)
        
        if not Path(self.lmdb_path).exists():
            raise FileNotFoundError(f"LMDB folder not found: {self.lmdb_path}")
        
        # Get total count
        temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with temp_env.begin() as txn:
            stat = txn.stat()
            self.length = stat['entries'] // 2  # Half are images, half are labels
        temp_env.close()
        
        self._env = None
        
        # SCD is medium-sized, no chunking needed
        self.chunk_start = 0
        self.chunk_end = self.length
        self.chunk_length = self.length
        
        print(f"SCD {split}: {self.length} samples")
        
        # Initialize preprocessor and augmentation
        self.preprocessor = DocumentPreprocessor(config, self.dataset_name)
        self.augmentation = DatasetAwareAugmentation(
            config,
            self.dataset_name,
            is_training=(split == 'train')
        )


def get_dataset(config, dataset_name: str, split: str = 'train', **kwargs) -> Dataset:
    """
    Factory function to get dataset
    
    Args:
        config: Configuration object
        dataset_name: Dataset name
        split: Data split
        **kwargs: Additional arguments (e.g., chunk_start, chunk_end)
    
    Returns:
        Dataset instance
    """
    if dataset_name == 'doctamper':
        return DocTamperDataset(config, split, **kwargs)
    elif dataset_name == 'rtm':
        return RTMDataset(config, split)
    elif dataset_name == 'casia':
        return CASIADataset(config, split)
    elif dataset_name == 'receipts':
        return ReceiptsDataset(config, split)
    elif dataset_name == 'fcd':
        return FCDDataset(config, split)
    elif dataset_name == 'scd':
        return SCDDataset(config, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
