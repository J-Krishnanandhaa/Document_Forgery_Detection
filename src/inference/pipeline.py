"""
Inference pipeline for document forgery detection
Complete pipeline: Image → Localization → Regions → Classification → Output
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from PIL import Image
import fitz  # PyMuPDF

from ..config import get_config
from ..models import get_model
from ..features import (
    get_feature_extractor, 
    get_mask_refiner, 
    get_region_extractor
)
from ..training.classifier import get_classifier


class ForgeryDetectionPipeline:
    """
    Complete inference pipeline for document forgery detection
    
    Pipeline:
    1. Input handling (PDF/Image)
    2. Preprocessing
    3. Deep localization
    4. Mask refinement
    5. Region extraction
    6. Feature extraction
    7. Classification
    8. Post-processing
    9. Output generation
    """
    
    def __init__(self, 
                 config,
                 model_path: str,
                 classifier_path: Optional[str] = None,
                 is_text_document: bool = True):
        """
        Initialize pipeline
        
        Args:
            config: Configuration object
            model_path: Path to localization model checkpoint
            classifier_path: Path to classifier (optional)
            is_text_document: Whether input is text document (for OCR features)
        """
        self.config = config
        self.is_text_document = is_text_document
        
        # Device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config.get('system.device') == 'cuda'
            else 'cpu'
        )
        print(f"Inference device: {self.device}")
        
        # Load localization model
        self.model = get_model(config).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        
        # Initialize mask refiner
        self.mask_refiner = get_mask_refiner(config, 'default')
        
        # Initialize region extractor
        self.region_extractor = get_region_extractor(config, 'default')
        
        # Initialize feature extractor
        self.feature_extractor = get_feature_extractor(config, is_text_document)
        
        # Load classifier if provided
        if classifier_path:
            self.classifier = get_classifier(config)
            self.classifier.load(classifier_path)
        else:
            self.classifier = None
        
        # Confidence threshold
        self.confidence_threshold = config.get('classifier.confidence_threshold', 0.6)
        
        # Image size
        self.image_size = config.get('data.image_size', 384)
        
        print("Inference pipeline initialized")
    
    def _load_model(self, model_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded model from {model_path}")
    
    def _load_image(self, input_path: str) -> np.ndarray:
        """
        Load image from file or PDF
        
        Args:
            input_path: Path to image or PDF
        
        Returns:
            Image as numpy array (H, W, 3)
        """
        path = Path(input_path)
        
        if path.suffix.lower() == '.pdf':
            # Rasterize PDF at 300 DPI
            doc = fitz.open(str(path))
            page = doc[0]
            mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
            pix = page.get_pixmap(matrix=mat)
            image = np.frombuffer(pix.samples, dtype=np.uint8)
            image = image.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            doc.close()
        else:
            # Load image
            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            Preprocessed image and original image
        """
        original = image.copy()
        
        # Resize
        preprocessed = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        preprocessed = preprocessed.astype(np.float32) / 255.0
        
        return preprocessed, original
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor"""
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def run(self, 
            input_path: str,
            output_dir: Optional[str] = None) -> Dict:
        """
        Run full inference pipeline
        
        Args:
            input_path: Path to input image or PDF
            output_dir: Optional output directory
        
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"{'='*60}")
        
        # 1. Load image
        image = self._load_image(input_path)
        original_size = image.shape[:2]
        print(f"Input size: {original_size}")
        
        # 2. Preprocess
        preprocessed, original = self._preprocess(image)
        tensor = self._to_tensor(preprocessed)
        
        # 3. Deep localization
        with torch.no_grad():
            logits, decoder_features = self.model(tensor)
            probability_map = torch.sigmoid(logits).cpu().numpy()[0, 0]
        
        print(f"Localization complete. Max prob: {probability_map.max():.3f}")
        
        # 4. Mask refinement
        binary_mask = self.mask_refiner.refine(probability_map, original_size)
        num_positive_pixels = binary_mask.sum()
        print(f"Mask refinement: {num_positive_pixels} positive pixels")
        
        # 5. Region extraction
        # Resize probability map to original size for confidence aggregation
        prob_resized = cv2.resize(probability_map, (original_size[1], original_size[0]))
        
        regions = self.region_extractor.extract(binary_mask, prob_resized, original)
        print(f"Regions extracted: {len(regions)}")
        
        # 6. Feature extraction & 7. Classification
        results = []
        
        for region in regions:
            # Extract features
            features = self.feature_extractor.extract(
                preprocessed,
                cv2.resize(region['region_mask'], (self.image_size, self.image_size)),
                [f.cpu() for f in decoder_features]
            )
            
            # Classify if classifier available
            if self.classifier is not None:
                predictions, confidences, valid_mask = self.classifier.predict_with_filtering(
                    features.reshape(1, -1)
                )
                
                if valid_mask[0]:
                    region['forgery_type'] = self.classifier.get_class_name(predictions[0])
                    region['classification_confidence'] = float(confidences[0])
                else:
                    # Low confidence - discard
                    continue
            else:
                region['forgery_type'] = 'unknown'
                region['classification_confidence'] = region['confidence']
            
            # Clean up non-serializable fields
            region_result = {
                'region_id': region['region_id'],
                'bounding_box': region['bounding_box'],
                'forgery_type': region['forgery_type'],
                'confidence': region['confidence'],
                'classification_confidence': region['classification_confidence'],
                'mask_probability_mean': region['mask_probability_mean'],
                'area': region['area']
            }
            results.append(region_result)
        
        print(f"Valid regions after filtering: {len(results)}")
        
        # 8. Post-processing - False positive removal
        results = self._post_process(results)
        
        # 9. Generate output
        output = {
            'input_path': str(input_path),
            'original_size': original_size,
            'num_regions': len(results),
            'regions': results,
            'is_tampered': len(results) > 0
        }
        
        # Save outputs if directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            input_name = Path(input_path).stem
            
            # Save final mask
            mask_path = output_path / f'{input_name}_mask.png'
            cv2.imwrite(str(mask_path), binary_mask * 255)
            
            # Save overlay visualization
            overlay = self._create_overlay(original, binary_mask, results)
            overlay_path = output_path / f'{input_name}_overlay.png'
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Save JSON
            json_path = output_path / f'{input_name}_results.json'
            with open(json_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\nOutputs saved to: {output_path}")
            output['mask_path'] = str(mask_path)
            output['overlay_path'] = str(overlay_path)
            output['json_path'] = str(json_path)
        
        return output
    
    def _post_process(self, regions: List[Dict]) -> List[Dict]:
        """
        Post-process regions to remove false positives
        
        Args:
            regions: List of region dictionaries
        
        Returns:
            Filtered regions
        """
        filtered = []
        
        for region in regions:
            # Confidence filtering
            if region['confidence'] < self.confidence_threshold:
                continue
            
            filtered.append(region)
        
        return filtered
    
    def _create_overlay(self, 
                       image: np.ndarray, 
                       mask: np.ndarray,
                       regions: List[Dict]) -> np.ndarray:
        """
        Create visualization overlay
        
        Args:
            image: Original image
            mask: Binary mask
            regions: Detected regions
        
        Returns:
            Overlay image
        """
        overlay = image.copy()
        alpha = self.config.get('outputs.visualization.overlay_alpha', 0.5)
        
        # Create colored mask
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [255, 0, 0]  # Red for forgery
        
        # Blend
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        overlay = np.where(
            mask_resized[:, :, None] > 0,
            (1 - alpha) * image + alpha * mask_colored,
            image
        ).astype(np.uint8)
        
        # Draw bounding boxes and labels
        for region in regions:
            x, y, w, h = region['bounding_box']
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{region['forgery_type']} ({region['confidence']:.2f})"
            cv2.putText(overlay, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return overlay


def get_pipeline(config, 
                model_path: str,
                classifier_path: Optional[str] = None,
                is_text_document: bool = True) -> ForgeryDetectionPipeline:
    """Factory function for pipeline"""
    return ForgeryDetectionPipeline(config, model_path, classifier_path, is_text_document)
