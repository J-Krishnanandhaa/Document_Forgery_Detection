"""
Hybrid feature extraction for forgery detection
Implements Critical Fix #5: Feature Group Gating
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from scipy import ndimage
from scipy.fftpack import dct
import pywt
from skimage.measure import regionprops, label
from skimage.filters import sobel


class DeepFeatureExtractor:
    """Extract deep features from decoder feature maps"""
    
    def __init__(self):
        """Initialize deep feature extractor"""
        pass
    
    def extract(self, 
                decoder_features: List[torch.Tensor],
                region_mask: np.ndarray) -> np.ndarray:
        """
        Extract deep features using Global Average Pooling
        
        Args:
            decoder_features: List of decoder feature tensors
            region_mask: Binary region mask (H, W)
        
        Returns:
            Deep feature vector
        """
        features = []
        
        for feat in decoder_features:
            # Ensure on CPU and numpy
            if isinstance(feat, torch.Tensor):
                feat = feat.detach().cpu().numpy()
            
            # feat shape: (B, C, H, W) or (C, H, W)
            if feat.ndim == 4:
                feat = feat[0]  # Take first batch
            
            # Resize mask to feature size
            h, w = feat.shape[1:]
            mask_resized = cv2.resize(region_mask.astype(np.float32), (w, h))
            mask_resized = mask_resized > 0.5
            
            # Masked Global Average Pooling
            if mask_resized.sum() > 0:
                for c in range(feat.shape[0]):
                    channel_feat = feat[c]
                    masked_mean = channel_feat[mask_resized].mean()
                    features.append(masked_mean)
            else:
                # Fallback: use global average
                features.extend(feat.mean(axis=(1, 2)).tolist())
        
        return np.array(features, dtype=np.float32)


class StatisticalFeatureExtractor:
    """Extract statistical and shape features from regions"""
    
    def __init__(self):
        """Initialize statistical feature extractor"""
        pass
    
    def extract(self, 
                image: np.ndarray,
                region_mask: np.ndarray) -> np.ndarray:
        """
        Extract statistical and shape features
        
        Args:
            image: Input image (H, W, 3) normalized [0, 1]
            region_mask: Binary region mask (H, W)
        
        Returns:
            Statistical feature vector
        """
        features = []
        
        # Label the mask
        labeled_mask = label(region_mask)
        props = regionprops(labeled_mask)
        
        if len(props) > 0:
            prop = props[0]
            
            # Area and perimeter
            features.append(prop.area)
            features.append(prop.perimeter)
            
            # Aspect ratio
            if prop.major_axis_length > 0:
                aspect_ratio = prop.minor_axis_length / prop.major_axis_length
            else:
                aspect_ratio = 1.0
            features.append(aspect_ratio)
            
            # Solidity
            features.append(prop.solidity)
            
            # Eccentricity
            features.append(prop.eccentricity)
            
            # Entropy (using intensity)
            if len(image.shape) == 3:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (image * 255).astype(np.uint8)
            
            # Resize region_mask to match gray image dimensions
            if region_mask.shape != gray.shape:
                region_mask_resized = cv2.resize(
                    region_mask.astype(np.uint8),
                    (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                region_mask_resized = region_mask
            
            region_pixels = gray[region_mask_resized > 0]
            if len(region_pixels) > 0:
                hist, _ = np.histogram(region_pixels, bins=256, range=(0, 256))
                hist = hist / hist.sum() + 1e-8
                entropy = -np.sum(hist * np.log2(hist + 1e-8))
            else:
                entropy = 0.0
            features.append(entropy)
        else:
            # Default values
            features.extend([0, 0, 1.0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)


class FrequencyFeatureExtractor:
    """Extract frequency-domain features"""
    
    def __init__(self):
        """Initialize frequency feature extractor"""
        pass
    
    def extract(self, 
                image: np.ndarray,
                region_mask: np.ndarray) -> np.ndarray:
        """
        Extract frequency-domain features (DCT, wavelet)
        
        Args:
            image: Input image (H, W, 3) normalized [0, 1]
            region_mask: Binary region mask (H, W)
        
        Returns:
            Frequency feature vector
        """
        features = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Get region bounding box
        coords = np.where(region_mask > 0)
        if len(coords[0]) == 0:
            return np.zeros(13, dtype=np.float32)
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Crop region
        region = gray[y_min:y_max+1, x_min:x_max+1].astype(np.float32)
        
        if region.size == 0:
            return np.zeros(13, dtype=np.float32)
        
        # DCT coefficients
        try:
            dct_coeffs = dct(dct(region, axis=0, norm='ortho'), axis=1, norm='ortho')
            
            # Mean and std of DCT coefficients
            features.append(np.mean(np.abs(dct_coeffs)))
            features.append(np.std(dct_coeffs))
            
            # High-frequency energy (bottom-right quadrant)
            h, w = dct_coeffs.shape
            high_freq = dct_coeffs[h//2:, w//2:]
            features.append(np.sum(np.abs(high_freq)) / (high_freq.size + 1e-8))
        except Exception:
            features.extend([0, 0, 0])
        
        # Wavelet features
        try:
            coeffs = pywt.dwt2(region, 'db1')
            cA, (cH, cV, cD) = coeffs
            
            # Energy in each sub-band
            features.append(np.sum(cA ** 2) / (cA.size + 1e-8))
            features.append(np.sum(cH ** 2) / (cH.size + 1e-8))
            features.append(np.sum(cV ** 2) / (cV.size + 1e-8))
            features.append(np.sum(cD ** 2) / (cD.size + 1e-8))
            
            # Wavelet entropy
            for coeff in [cH, cV, cD]:
                coeff_flat = np.abs(coeff.flatten())
                if coeff_flat.sum() > 0:
                    coeff_norm = coeff_flat / coeff_flat.sum()
                    entropy = -np.sum(coeff_norm * np.log2(coeff_norm + 1e-8))
                else:
                    entropy = 0.0
                features.append(entropy)
        except Exception:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)


class NoiseELAFeatureExtractor:
    """Extract noise and Error Level Analysis features"""
    
    def __init__(self, quality: int = 90):
        """
        Initialize noise/ELA extractor
        
        Args:
            quality: JPEG quality for ELA
        """
        self.quality = quality
    
    def extract(self, 
                image: np.ndarray,
                region_mask: np.ndarray) -> np.ndarray:
        """
        Extract noise and ELA features
        
        Args:
            image: Input image (H, W, 3) normalized [0, 1]
            region_mask: Binary region mask (H, W)
        
        Returns:
            Noise/ELA feature vector
        """
        features = []
        
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Error Level Analysis
        # Compress and compute difference
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        _, encoded = cv2.imencode('.jpg', img_uint8, encode_param)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        ela = np.abs(img_uint8.astype(np.float32) - recompressed.astype(np.float32))
        
        # ELA features within region
        # Resize region_mask to match ela dimensions
        if region_mask.shape[:2] != ela.shape[:2]:
            mask_resized = cv2.resize(
                region_mask.astype(np.uint8),
                (ela.shape[1], ela.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = region_mask
        
        ela_region = ela[mask_resized > 0]
        if len(ela_region) > 0:
            features.append(np.mean(ela_region))  # ELA mean
            features.append(np.var(ela_region))   # ELA variance
            features.append(np.max(ela_region))   # ELA max
        else:
            features.extend([0, 0, 0])
        
        # Noise residual (using median filter)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_uint8
        
        median_filtered = cv2.medianBlur(gray, 3)
        noise_residual = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
        
        # Resize region_mask to match noise_residual dimensions
        if region_mask.shape != noise_residual.shape:
            mask_resized = cv2.resize(
                region_mask.astype(np.uint8),
                (noise_residual.shape[1], noise_residual.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = region_mask
        
        residual_region = noise_residual[mask_resized > 0]
        if len(residual_region) > 0:
            features.append(np.mean(residual_region))
            features.append(np.var(residual_region))
        else:
            features.extend([0, 0])
        
        return np.array(features, dtype=np.float32)


class OCRFeatureExtractor:
    """
    Extract OCR-based consistency features
    Only for text documents (Feature Gating - Critical Fix #5)
    """
    
    def __init__(self):
        """Initialize OCR feature extractor"""
        self.ocr_available = False
        
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=True)
            self.ocr_available = True
        except Exception:
            print("Warning: EasyOCR not available, OCR features disabled")
    
    def extract(self, 
                image: np.ndarray,
                region_mask: np.ndarray) -> np.ndarray:
        """
        Extract OCR consistency features
        
        Args:
            image: Input image (H, W, 3) normalized [0, 1]
            region_mask: Binary region mask (H, W)
        
        Returns:
            OCR feature vector (or zeros if not text document)
        """
        features = []
        
        if not self.ocr_available:
            return np.zeros(6, dtype=np.float32)
        
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Get region bounding box
        coords = np.where(region_mask > 0)
        if len(coords[0]) == 0:
            return np.zeros(6, dtype=np.float32)
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Crop region
        region = img_uint8[y_min:y_max+1, x_min:x_max+1]
        
        try:
            # OCR on region
            results = self.reader.readtext(region)
            
            if len(results) > 0:
                # Confidence deviation
                confidences = [r[2] for r in results]
                features.append(np.mean(confidences))
                features.append(np.std(confidences))
                
                # Character spacing analysis
                bbox_widths = [abs(r[0][1][0] - r[0][0][0]) for r in results]
                if len(bbox_widths) > 1:
                    features.append(np.std(bbox_widths) / (np.mean(bbox_widths) + 1e-8))
                else:
                    features.append(0.0)
                
                # Text density
                features.append(len(results) / (region.shape[0] * region.shape[1] + 1e-8))
                
                # Stroke width variation (using edge detection)
                gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                edges = sobel(gray_region)
                features.append(np.mean(edges))
                features.append(np.std(edges))
            else:
                features.extend([0, 0, 0, 0, 0, 0])
        except Exception:
            features.extend([0, 0, 0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)


class HybridFeatureExtractor:
    """
    Complete hybrid feature extraction
    Implements Critical Fix #5: Feature Group Gating
    """
    
    def __init__(self, config, is_text_document: bool = True):
        """
        Initialize hybrid feature extractor
        
        Args:
            config: Configuration object
            is_text_document: Whether input is text document (for OCR gating)
        """
        self.config = config
        self.is_text_document = is_text_document
        
        # Initialize extractors
        self.deep_extractor = DeepFeatureExtractor()
        self.stat_extractor = StatisticalFeatureExtractor()
        self.freq_extractor = FrequencyFeatureExtractor()
        self.noise_extractor = NoiseELAFeatureExtractor()
        
        # Critical Fix #5: OCR only for text documents
        if is_text_document and config.get('features.ocr.enabled', True):
            self.ocr_extractor = OCRFeatureExtractor()
        else:
            self.ocr_extractor = None
    
    def extract(self,
                image: np.ndarray,
                region_mask: np.ndarray,
                decoder_features: Optional[List[torch.Tensor]] = None) -> np.ndarray:
        """
        Extract all hybrid features for a region
        
        Args:
            image: Input image (H, W, 3) normalized [0, 1]
            region_mask: Binary region mask (H, W)
            decoder_features: Optional decoder features for deep feature extraction
        
        Returns:
            Concatenated feature vector
        """
        all_features = []
        
        # Deep features (if available)
        if decoder_features is not None and self.config.get('features.deep.enabled', True):
            deep_feats = self.deep_extractor.extract(decoder_features, region_mask)
            all_features.append(deep_feats)
        
        # Statistical & shape features
        if self.config.get('features.statistical.enabled', True):
            stat_feats = self.stat_extractor.extract(image, region_mask)
            all_features.append(stat_feats)
        
        # Frequency-domain features
        if self.config.get('features.frequency.enabled', True):
            freq_feats = self.freq_extractor.extract(image, region_mask)
            all_features.append(freq_feats)
        
        # Noise & ELA features
        if self.config.get('features.noise.enabled', True):
            noise_feats = self.noise_extractor.extract(image, region_mask)
            all_features.append(noise_feats)
        
        # Critical Fix #5: OCR features only for text documents
        if self.ocr_extractor is not None:
            ocr_feats = self.ocr_extractor.extract(image, region_mask)
            all_features.append(ocr_feats)
        
        # Concatenate all features
        if len(all_features) > 0:
            features = np.concatenate(all_features)
        else:
            features = np.array([], dtype=np.float32)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for interpretability"""
        names = []
        
        if self.config.get('features.deep.enabled', True):
            names.extend([f'deep_{i}' for i in range(256)])  # Approximate
        
        if self.config.get('features.statistical.enabled', True):
            names.extend(['area', 'perimeter', 'aspect_ratio', 
                         'solidity', 'eccentricity', 'entropy'])
        
        if self.config.get('features.frequency.enabled', True):
            names.extend(['dct_mean', 'dct_std', 'high_freq_energy',
                         'wavelet_cA', 'wavelet_cH', 'wavelet_cV', 'wavelet_cD',
                         'wavelet_entropy_H', 'wavelet_entropy_V', 'wavelet_entropy_D'])
        
        if self.config.get('features.noise.enabled', True):
            names.extend(['ela_mean', 'ela_var', 'ela_max',
                         'noise_residual_mean', 'noise_residual_var'])
        
        if self.ocr_extractor is not None:
            names.extend(['ocr_conf_mean', 'ocr_conf_std', 'spacing_irregularity',
                         'text_density', 'stroke_mean', 'stroke_std'])
        
        return names


def get_feature_extractor(config, is_text_document: bool = True) -> HybridFeatureExtractor:
    """
    Factory function to create feature extractor
    
    Args:
        config: Configuration object
        is_text_document: Whether input is text document
    
    Returns:
        HybridFeatureExtractor instance
    """
    return HybridFeatureExtractor(config, is_text_document)
