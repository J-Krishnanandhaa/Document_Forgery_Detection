"""
Mask refinement and region extraction
Implements Critical Fix #3: Adaptive Mask Refinement Thresholds
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
from skimage.measure import label, regionprops


class MaskRefiner:
    """
    Mask refinement with adaptive thresholds
    Implements Critical Fix #3: Dataset-specific minimum region areas
    """
    
    def __init__(self, config, dataset_name: str = 'default'):
        """
        Initialize mask refiner
        
        Args:
            config: Configuration object
            dataset_name: Dataset name for adaptive thresholds
        """
        self.config = config
        self.dataset_name = dataset_name
        
        # Get mask refinement parameters
        self.threshold = config.get('mask_refinement.threshold', 0.5)
        self.closing_kernel = config.get('mask_refinement.morphology.closing_kernel', 5)
        self.opening_kernel = config.get('mask_refinement.morphology.opening_kernel', 3)
        
        # Critical Fix #3: Adaptive thresholds per dataset
        self.min_region_area = config.get_min_region_area(dataset_name)
        
        print(f"MaskRefiner initialized for {dataset_name}")
        print(f"Min region area: {self.min_region_area * 100:.2f}%")
    
    def refine(self, 
               probability_map: np.ndarray,
               original_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Refine probability map to binary mask
        
        Args:
            probability_map: Forgery probability map (H, W), values [0, 1]
            original_size: Optional (H, W) to resize mask back to original
        
        Returns:
            Refined binary mask (H, W)
        """
        # Threshold to binary
        binary_mask = (probability_map > self.threshold).astype(np.uint8)
        
        # Morphological closing (fill broken strokes)
        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.closing_kernel, self.closing_kernel)
        )
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)
        
        # Morphological opening (remove isolated noise)
        opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.opening_kernel, self.opening_kernel)
        )
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, opening_kernel)
        
        # Critical Fix #3: Remove small regions with adaptive threshold
        binary_mask = self._remove_small_regions(binary_mask)
        
        # Resize to original size if provided
        if original_size is not None:
            binary_mask = cv2.resize(
                binary_mask, 
                (original_size[1], original_size[0]),  # cv2 uses (W, H)
                interpolation=cv2.INTER_NEAREST
            )
        
        return binary_mask
    
    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove regions smaller than minimum area threshold
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            Filtered mask
        """
        # Calculate minimum pixel count
        image_area = mask.shape[0] * mask.shape[1]
        min_pixels = int(image_area * self.min_region_area)
        
        # Label connected components
        labeled_mask, num_features = ndimage.label(mask)
        
        # Keep only large enough regions
        filtered_mask = np.zeros_like(mask)
        
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_mask == region_id)
            region_area = region_mask.sum()
            
            if region_area >= min_pixels:
                filtered_mask[region_mask] = 1
        
        return filtered_mask


class RegionExtractor:
    """
    Extract individual regions from binary mask
    Implements Critical Fix #4: Region Confidence Aggregation
    """
    
    def __init__(self, config, dataset_name: str = 'default'):
        """
        Initialize region extractor
        
        Args:
            config: Configuration object
            dataset_name: Dataset name
        """
        self.config = config
        self.dataset_name = dataset_name
        self.min_region_area = config.get_min_region_area(dataset_name)
    
    def extract(self, 
                binary_mask: np.ndarray,
                probability_map: np.ndarray,
                original_image: np.ndarray) -> List[Dict]:
        """
        Extract regions from binary mask
        
        Args:
            binary_mask: Refined binary mask (H, W)
            probability_map: Original probability map (H, W)
            original_image: Original image (H, W, 3)
        
        Returns:
            List of region dictionaries with bounding box, mask, image, confidence
        """
        regions = []
        
        print(f"[REGION_EXTRACT] Input shapes:")
        print(f"  - binary_mask: {binary_mask.shape}")
        print(f"  - probability_map: {probability_map.shape}")
        print(f"  - original_image: {original_image.shape}")
        
        # Safety check: Ensure probability_map and binary_mask have same dimensions
        if probability_map.shape != binary_mask.shape:
            print(f"[REGION_EXTRACT] WARNING: Shape mismatch! Resizing probability_map from {probability_map.shape} to {binary_mask.shape}")
            import cv2
            probability_map = cv2.resize(
                probability_map,
                (binary_mask.shape[1], binary_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            print(f"[REGION_EXTRACT] After resize: probability_map shape = {probability_map.shape}")
        
        # Connected component analysis (8-connectivity)
        labeled_mask = label(binary_mask, connectivity=2)
        props = regionprops(labeled_mask)
        
        for region_id, prop in enumerate(props, start=1):
            # Bounding box
            y_min, x_min, y_max, x_max = prop.bbox
            
            # Region mask
            region_mask = (labeled_mask == region_id).astype(np.uint8)
            
            # Cropped region image
            region_image = original_image[y_min:y_max, x_min:x_max].copy()
            region_mask_cropped = region_mask[y_min:y_max, x_min:x_max]
            
            
            # Critical Fix #4: Region-level confidence aggregation
            # Ensure region_mask and probability_map have same shape
            if region_mask.shape != probability_map.shape:
                import cv2
                # Resize probability_map to match region_mask
                probability_map = cv2.resize(
                    probability_map,
                    (region_mask.shape[1], region_mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            
            region_probs = probability_map[region_mask > 0]
            region_confidence = float(np.mean(region_probs)) if len(region_probs) > 0 else 0.0
            
            regions.append({
                'region_id': region_id,
                'bounding_box': [int(x_min), int(y_min), 
                               int(x_max - x_min), int(y_max - y_min)],
                'area': prop.area,
                'centroid': (int(prop.centroid[1]), int(prop.centroid[0])),
                'region_mask': region_mask,
                'region_mask_cropped': region_mask_cropped,
                'region_image': region_image,
                'confidence': region_confidence,
                'mask_probability_mean': region_confidence
            })
        
        return regions
    
    def extract_for_casia(self, 
                          binary_mask: np.ndarray,
                          probability_map: np.ndarray,
                          original_image: np.ndarray) -> List[Dict]:
        """
        Critical Fix #6: CASIA handling - treat entire image as one region
        
        Args:
            binary_mask: Binary mask (may be empty for authentic images)
            probability_map: Probability map
            original_image: Original image
        
        Returns:
            Single region representing entire image
        """
        h, w = original_image.shape[:2]
        
        # Create single region covering entire image
        region_mask = np.ones((h, w), dtype=np.uint8)
        
        # Overall confidence from probability map
        overall_confidence = float(np.mean(probability_map))
        
        return [{
            'region_id': 1,
            'bounding_box': [0, 0, w, h],
            'area': h * w,
            'centroid': (w // 2, h // 2),
            'region_mask': region_mask,
            'region_mask_cropped': region_mask,
            'region_image': original_image,
            'confidence': overall_confidence,
            'mask_probability_mean': overall_confidence
        }]


def get_mask_refiner(config, dataset_name: str = 'default') -> MaskRefiner:
    """Factory function for mask refiner"""
    return MaskRefiner(config, dataset_name)


def get_region_extractor(config, dataset_name: str = 'default') -> RegionExtractor:
    """Factory function for region extractor"""
    return RegionExtractor(config, dataset_name)
