"""
Dataset-aware preprocessing for document forgery detection
Implements Critical Fix #1: Dataset-Aware Preprocessing
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import pywt
from scipy import ndimage


class DocumentPreprocessor:
    """Dataset-aware document preprocessing"""
    
    def __init__(self, config, dataset_name: str):
        """
        Initialize preprocessor
        
        Args:
            config: Configuration object
            dataset_name: Name of dataset (for dataset-aware processing)
        """
        self.config = config
        self.dataset_name = dataset_name
        self.image_size = config.get('data.image_size', 384)
        self.noise_threshold = config.get('preprocessing.noise_threshold', 15.0)
        
        # Dataset-aware flags (Critical Fix #1)
        self.skip_deskew = config.should_skip_deskew(dataset_name)
        self.skip_denoising = config.should_skip_denoising(dataset_name)
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply preprocessing pipeline
        
        Args:
            image: Input image (H, W, 3)
            mask: Optional ground truth mask (H, W)
        
        Returns:
            Preprocessed image and mask
        """
        # 1. Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Deskew (dataset-aware)
        if not self.skip_deskew:
            image, mask = self._deskew(image, mask)
        
        # 3. Resize
        image, mask = self._resize(image, mask)
        
        # 4. Normalize
        image = self._normalize(image)
        
        # 5. Conditional denoising (dataset-aware)
        if not self.skip_denoising:
            noise_level = self._estimate_noise(image)
            if noise_level > self.noise_threshold:
                image = self._denoise(image)
        
        return image, mask
    
    def _deskew(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Deskew document image
        
        Args:
            image: Input image
            mask: Optional mask
        
        Returns:
            Deskewed image and mask
        """
        # Convert to grayscale for angle detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None and len(lines) > 0:
            # Calculate dominant angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                angles.append(angle)
            
            # Use median angle
            angle = np.median(angles)
            
            # Only deskew if angle is significant (> 0.5 degrees)
            if abs(angle) > 0.5:
                # Get rotation matrix
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Rotate image
                image = cv2.warpAffine(image, M, (w, h), 
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
                
                # Rotate mask if provided
                if mask is not None:
                    mask = cv2.warpAffine(mask, M, (w, h),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
        
        return image, mask
    
    def _resize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Resize image and mask to target size
        
        Args:
            image: Input image
            mask: Optional mask
        
        Returns:
            Resized image and mask
        """
        target_size = (self.image_size, self.image_size)
        
        # Resize image
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Resize mask if provided
        if mask is not None:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        return image, mask
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1]
        
        Args:
            image: Input image
        
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise level using Laplacian variance and wavelet-based estimation
        
        Args:
            image: Input image (normalized)
        
        Returns:
            Estimated noise level
        """
        # Convert to grayscale for noise estimation
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Method 1: Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Method 2: Wavelet-based noise estimation
        coeffs = pywt.dwt2(gray, 'db1')
        _, (cH, cV, cD) = coeffs
        sigma = np.median(np.abs(cD)) / 0.6745
        
        # Combine both estimates
        noise_level = (laplacian_var + sigma) / 2.0
        
        return noise_level
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply conditional denoising
        
        Args:
            image: Input image (normalized)
        
        Returns:
            Denoised image
        """
        # Convert to uint8 for filtering
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply median filter (3x3)
        median_filtered = cv2.medianBlur(image_uint8, 3)
        
        # Apply Gaussian filter (σ ≤ 0.8)
        gaussian_filtered = cv2.GaussianBlur(median_filtered, (3, 3), 0.8)
        
        # Convert back to float32
        denoised = gaussian_filtered.astype(np.float32) / 255.0
        
        return denoised


def preprocess_image(image: np.ndarray, 
                     mask: Optional[np.ndarray] = None,
                     config = None,
                     dataset_name: str = 'default') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function for preprocessing
    
    Args:
        image: Input image
        mask: Optional mask
        config: Configuration object
        dataset_name: Dataset name
    
    Returns:
        Preprocessed image and mask
    """
    preprocessor = DocumentPreprocessor(config, dataset_name)
    return preprocessor(image, mask)
