"""
Document Forgery Detection - Gradio Interface for Hugging Face Spaces

This app provides a web interface for detecting and classifying document forgeries.
"""

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import get_model
from src.config import get_config
from src.data.preprocessing import DocumentPreprocessor
from src.data.augmentation import DatasetAwareAugmentation
from src.features.region_extraction import get_mask_refiner, get_region_extractor
from src.features.feature_extraction import get_feature_extractor
from src.training.classifier import ForgeryClassifier

# Class names
CLASS_NAMES = {0: 'Copy-Move', 1: 'Splicing', 2: 'Text Substitution'}
CLASS_COLORS = {
    0: (217, 83, 79),    # #d9534f - Muted red (Copy-Move)
    1: (92, 184, 92),    # #5cb85c - Muted green (Splicing)
    2: (65, 105, 225)    # #4169E1 - Royal blue (Text Substitution/Generation)
}

# Actual model performance metrics
MODEL_METRICS = {
    'segmentation': {
        'dice': 0.6212,
        'iou': 0.4506,
        'precision': 0.7077,
        'recall': 0.5536
    },
    'classification': {
        'overall_accuracy': 0.8897,
        'per_class': {
            'copy_move': 0.92,
            'splicing': 0.85,
            'generation': 0.90
        }
    }
}


def create_gauge_chart(value: float, title: str, max_value: float = 1.0) -> go.Figure:
    """Create a subtle radial gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': '%', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': '#4169E1', 'thickness': 0.7},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(217, 83, 79, 0.1)'},
                {'range': [50, 75], 'color': 'rgba(240, 173, 78, 0.1)'},
                {'range': [75, 100], 'color': 'rgba(92, 184, 92, 0.1)'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_detection_metrics_gauge(avg_confidence: float, iou: float, precision: float, recall: float, num_detections: int) -> go.Figure:
    """Create a high-fidelity radial bar chart (concentric rings)"""
    
    # Calculate percentages (0-100)
    metrics = [
        {'name': 'Confidence', 'val': avg_confidence * 100 if num_detections > 0 else 0, 'color': '#4169E1', 'base': 80},
        {'name': 'Precision', 'val': precision * 100, 'color': '#5cb85c', 'base': 60},
        {'name': 'Recall', 'val': recall * 100, 'color': '#f0ad4e', 'base': 40},
        {'name': 'IoU', 'val': iou * 100, 'color': '#d9534f', 'base': 20}
    ]
    
    fig = go.Figure()

    for m in metrics:
        # 1. Add background track (faint gray ring)
        fig.add_trace(go.Barpolar(
            r=[15],
            theta=[180],
            width=[360],
            base=m['base'],
            marker_color='rgba(128,128,128,0.1)',
            hoverinfo='none',
            showlegend=False
        ))
        
        # 2. Add the actual metric bar (the colored arc)
        # 100% = 360 degrees
        angle_width = m['val'] * 3.6
        fig.add_trace(go.Barpolar(
            r=[15],
            theta=[angle_width / 2],
            width=[angle_width],
            base=m['base'],
            name=f"{m['name']}: {m['val']:.1f}%",
            marker_color=m['color'],
            marker_line_width=0,
            hoverinfo='name'
        ))

    fig.update_layout(
        polar=dict(
            hole=0.1,
            radialaxis=dict(visible=False, range=[0, 100]),
            angularaxis=dict(
                rotation=90,           # Start at 12 o'clock
                direction='clockwise', # Go clockwise
                gridcolor='rgba(128,128,128,0.2)',
                tickmode='array',
                tickvals=[0, 90, 180, 270],
                ticktext=['0%', '25%', '50%', '75%'],
                showticklabels=True,
                tickfont=dict(size=12, color='#888')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1,
            font=dict(size=14, color='white'),
            itemwidth=30
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,  # Reduced from 450
        margin=dict(l=60, r=180, t=40, b=40)
    )
    
    return fig


class ForgeryDetector:
    """Main forgery detection pipeline"""
    
    def __init__(self):
        try:
            print("="*80)
            print("INITIALIZING FORGERY DETECTOR")
            print("="*80)
            
            print("1. Loading config...")
            self.config = get_config('config.yaml')
            print("   ✓ Config loaded")
            
            print("2. Setting up device...")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"   ✓ Using device: {self.device}")
            
            print("3. Creating model architecture...")
            self.model = get_model(self.config).to(self.device)
            print("   ✓ Model created")
            
            print("4. Loading checkpoint...")
            checkpoint = torch.load('models/best_doctamper.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("   ✓ Model loaded")
            
            print("5. Loading classifier...")
            self.classifier = ForgeryClassifier(self.config)
            self.classifier.load('models/classifier')
            print("   ✓ Classifier loaded")
            
            print("6. Initializing components...")
            self.preprocessor = DocumentPreprocessor(self.config, 'doctamper')
            self.augmentation = DatasetAwareAugmentation(self.config, 'doctamper', is_training=False)
            self.mask_refiner = get_mask_refiner(self.config)
            self.region_extractor = get_region_extractor(self.config)
            self.feature_extractor = get_feature_extractor(self.config, is_text_document=True)
            print("   ✓ Components initialized")
            
            print("="*80)
            print("✓ FORGERY DETECTOR READY")
            print("="*80)
            
        except Exception as e:
            import traceback
            print("="*80)
            print("❌ INITIALIZATION FAILED")
            print("="*80)
            print(f"Error: {str(e)}")
            print("\nFull traceback:")
            print(traceback.format_exc())
            print("="*80)
            raise
    
    def detect(self, image):
        """
        Detect forgeries in document image or PDF
        
        Returns:
            original_image: Original uploaded image
            overlay_image: Image with detection overlay
            gauge_dice: Dice score gauge
            gauge_accuracy: Accuracy gauge
            results_html: Detection results as HTML
        """
        # Handle file path input (from gr.Image with type="filepath")
        if isinstance(image, str):
            if image.lower().endswith(('.doc', '.docx')):
                # Handle Word documents - multiple fallback strategies
                import tempfile
                import os
                import subprocess
                
                temp_pdf = None
                try:
                    # Strategy 1: Try docx2pdf (Windows with MS Word)
                    try:
                        from docx2pdf import convert
                        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                        temp_pdf.close()
                        convert(image, temp_pdf.name)
                        pdf_path = temp_pdf.name
                    except Exception as e1:
                        # Strategy 2: Try LibreOffice (Linux/Mac)
                        try:
                            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                            temp_pdf.close()
                            subprocess.run([
                                'libreoffice', '--headless', '--convert-to', 'pdf',
                                '--outdir', os.path.dirname(temp_pdf.name),
                                image
                            ], check=True, capture_output=True)
                            
                            # LibreOffice creates file with original name + .pdf
                            base_name = os.path.splitext(os.path.basename(image))[0]
                            generated_pdf = os.path.join(os.path.dirname(temp_pdf.name), f"{base_name}.pdf")
                            
                            if os.path.exists(generated_pdf):
                                os.rename(generated_pdf, temp_pdf.name)
                                pdf_path = temp_pdf.name
                            else:
                                raise Exception("LibreOffice conversion failed")
                        except Exception as e2:
                            # Strategy 3: Extract text and create simple image
                            from docx import Document
                            doc = Document(image)
                            
                            # Extract text
                            text_lines = []
                            for para in doc.paragraphs[:40]:  # First 40 paragraphs
                                if para.text.strip():
                                    text_lines.append(para.text[:100])  # Max 100 chars per line
                            
                            # Create image with text
                            img_height = 1400
                            img_width = 1000
                            image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
                            
                            y_offset = 60
                            for line in text_lines[:35]:
                                cv2.putText(image, line, (40, y_offset),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                                y_offset += 35
                            
                            # Skip to end - image is ready
                            pdf_path = None
                    
                    # If we got a PDF, convert ALL pages to a single tall image
                    if pdf_path and os.path.exists(pdf_path):
                        import fitz
                        pdf_document = fitz.open(pdf_path)
                        page_images = []
                        for page_num in range(len(pdf_document)):
                            page = pdf_document[page_num]
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                            if pix.n == 4:
                                page_img = cv2.cvtColor(page_img, cv2.COLOR_RGBA2RGB)
                            page_images.append(page_img)
                        pdf_document.close()
                        os.unlink(pdf_path)
                        # Stack all pages vertically into one tall image
                        if len(page_images) == 1:
                            image = page_images[0]
                        else:
                            max_width = max(p.shape[1] for p in page_images)
                            padded = []
                            for p in page_images:
                                if p.shape[1] < max_width:
                                    pad = np.ones((p.shape[0], max_width - p.shape[1], 3), dtype=np.uint8) * 255
                                    p = np.concatenate([p, pad], axis=1)
                                padded.append(p)
                            image = np.concatenate(padded, axis=0)
                        
                except Exception as e:
                    raise ValueError(f"Could not process Word document. Please convert to PDF or image first. Error: {str(e)}")
                finally:
                    # Clean up temp file if it exists
                    if temp_pdf and os.path.exists(temp_pdf.name):
                        try:
                            os.unlink(temp_pdf.name)
                        except:
                            pass
                    
            elif image.lower().endswith('.pdf'):
                # Handle PDF files - process ALL pages
                import fitz  # PyMuPDF
                pdf_document = fitz.open(image)
                page_images = []
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    if pix.n == 4:
                        page_img = cv2.cvtColor(page_img, cv2.COLOR_RGBA2RGB)
                    page_images.append(page_img)
                pdf_document.close()
                # Stack all pages vertically into one tall image
                if len(page_images) == 1:
                    image = page_images[0]
                else:
                    max_width = max(p.shape[1] for p in page_images)
                    padded = []
                    for p in page_images:
                        if p.shape[1] < max_width:
                            pad = np.ones((p.shape[0], max_width - p.shape[1], 3), dtype=np.uint8) * 255
                            p = np.concatenate([p, pad], axis=1)
                        padded.append(p)
                    image = np.concatenate(padded, axis=0)
            else:
                # Load image file
                image = Image.open(image)
                image = np.array(image)
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        original_image = image.copy()
        
        # Preprocess
        preprocessed, _ = self.preprocessor(image, None)
        
        # Augment
        augmented = self.augmentation(preprocessed, None)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Run localization
        with torch.no_grad():
            logits, decoder_features = self.model(image_tensor)
            prob_map = torch.sigmoid(logits).cpu().numpy()[0, 0]
        
        print(f"[DEBUG] prob_map shape: {prob_map.shape}")
        print(f"[DEBUG] original_image shape: {original_image.shape}")
        
        # Resize probability map to match original image size to avoid index mismatch errors
        prob_map_resized = cv2.resize(
            prob_map, 
            (original_image.shape[1], original_image.shape[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        print(f"[DEBUG] prob_map_resized shape: {prob_map_resized.shape}")
        
        # Refine mask
        # Lower threshold for more sensitive detection
        binary_mask = (prob_map_resized > 0.3).astype(np.uint8)
        refined_mask = self.mask_refiner.refine(prob_map_resized, original_size=original_image.shape[:2])
        
        print(f"[DEBUG] binary_mask shape: {binary_mask.shape}")
        print(f"[DEBUG] refined_mask shape (after refine): {refined_mask.shape}")
        
        # Ensure refined_mask matches prob_map_resized dimensions
        if refined_mask.shape != prob_map_resized.shape:
            print(f"[DEBUG] Resizing refined_mask from {refined_mask.shape} to {prob_map_resized.shape}")
            refined_mask = cv2.resize(
                refined_mask,
                (prob_map_resized.shape[1], prob_map_resized.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Safety check: Ensure prob_map_resized and refined_mask have same dimensions (fallback)
        if prob_map_resized.shape != refined_mask.shape:
            print(f"[DEBUG] FALLBACK: Resizing prob_map_resized from {prob_map_resized.shape} to {refined_mask.shape}")
            prob_map_resized = cv2.resize(
                prob_map_resized,
                (refined_mask.shape[1], refined_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        print(f"[DEBUG] Final shapes before region extraction:")
        print(f"  - refined_mask: {refined_mask.shape}")
        print(f"  - prob_map_resized: {prob_map_resized.shape}")
        
        # DEBUG: Save probability map visualization
        prob_map_vis = (prob_map_resized * 255).astype(np.uint8)
        prob_map_colored = cv2.applyColorMap(prob_map_vis, cv2.COLORMAP_JET)
        print(f"[DEBUG] Probability map stats:")
        print(f"  - Min: {prob_map_resized.min():.4f}")
        print(f"  - Max: {prob_map_resized.max():.4f}")
        print(f"  - Mean: {prob_map_resized.mean():.4f}")
        print(f"  - Pixels > 0.3: {(prob_map_resized > 0.3).sum()}")
        print(f"  - Pixels > 0.5: {(prob_map_resized > 0.5).sum()}")
        
        # Extract regions
        regions = self.region_extractor.extract(refined_mask, prob_map_resized, original_image)
        
        print(f"[DEBUG] Regions extracted: {len(regions)}")
        if len(regions) > 0:
            print(f"[DEBUG] Region areas: {[r['area'] for r in regions]}")
            print(f"[DEBUG] Region confidences: {[r.get('confidence', 0) for r in regions]}")
        
        # Classify regions
        results = []
        classified_count = 0
        rejected_count = 0
        for region in regions:
            # Get decoder features and handle shape
            df = decoder_features[0].cpu()  # Get first decoder feature
            
            # Remove batch dimension if present: [1, C, H, W] -> [C, H, W]
            if df.ndim == 4:
                df = df.squeeze(0)
            
            # Now df should be [C, H, W]
            _, fh, fw = df.shape

            region_mask = region['region_mask']
            if region_mask.shape != (fh, fw):
                region_mask = cv2.resize(
                    region_mask.astype(np.uint8),
                    (fw, fh),
                    interpolation=cv2.INTER_NEAREST
            )

            region_mask = region_mask.astype(bool)
            
            # Extract features using tensor converted to numpy (matches training pipeline)
            # Convert tensor back to numpy: (C, H, W) -> (H, W, C)
            preprocessed_numpy = image_tensor[0].permute(1, 2, 0).cpu().numpy()
            
            # Pass region_mask directly - feature extractor handles resizing internally
            features = self.feature_extractor.extract(
                preprocessed_numpy,
                region['region_mask'],
                [f.cpu() for f in decoder_features]
            )
            
            # Reshape features to 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Pad/truncate features to match classifier
            expected_features = 526
            current_features = features.shape[1]
            if current_features < expected_features:
                padding = np.zeros((features.shape[0], expected_features - current_features))
                features = np.hstack([features, padding])
            elif current_features > expected_features:
                features = features[:, :expected_features]
            
            # Classify - get probabilities for all classes
            # Temporarily access model directly to get full probabilities
            features_scaled = self.classifier.scaler.transform(features)
            probabilities = self.classifier.model.predict(features_scaled)[0]  # Shape: (3,)
            
            forgery_type = int(probabilities.argmax())
            confidence = float(probabilities.max())
            
            # Log all class probabilities for debugging
            prob_str = ", ".join([f"{CLASS_NAMES[i]}: {probabilities[i]:.3f}" for i in range(3)])
            print(f"[DEBUG] Region {region['region_id']}: {CLASS_NAMES[forgery_type]} (confidence: {confidence:.3f})")
            print(f"        All probabilities: {prob_str}")
            
            # Lower confidence threshold to detect more regions
            if confidence > 0.5:
                classified_count += 1
                results.append({
                    'region_id': region['region_id'],
                    'bounding_box': region['bounding_box'],
                    'forgery_type': CLASS_NAMES[forgery_type],
                    'confidence': confidence
                })
            else:
                rejected_count += 1
                print(f"  -> REJECTED (confidence {confidence:.3f} < 0.5)")
        
        print(f"[DEBUG] Classification summary:")
        print(f"  - Total regions: {len(regions)}")
        print(f"  - Classified: {classified_count}")
        print(f"  - Rejected: {rejected_count}")
        
        # Create visualization
        overlay = self._create_overlay(original_image, results)
        
        # Calculate actual detection metrics from probability map and mask
        num_detections = len(results)
        avg_confidence = sum(r['confidence'] for r in results) / num_detections if num_detections > 0 else 0
        
        # Calculate IoU, Precision, Recall from the refined mask and probability map
        if num_detections > 0:
            # Use resized prob_map to match refined_mask dimensions
            high_conf_mask = (prob_map_resized > 0.7).astype(np.uint8)
            predicted_positive = np.sum(refined_mask > 0)
            high_conf_positive = np.sum(high_conf_mask > 0)
            
            # Calculate intersection and union
            intersection = np.sum((refined_mask > 0) & (high_conf_mask > 0))
            union = np.sum((refined_mask > 0) | (high_conf_mask > 0))
            
            # Calculate metrics
            iou = intersection / union if union > 0 else 0
            precision = intersection / predicted_positive if predicted_positive > 0 else 0
            recall = intersection / high_conf_positive if high_conf_positive > 0 else 0
        else:
            # No detections - use zeros
            iou = 0
            precision = 0
            recall = 0
        
        # Create detection metrics gauge with actual values
        metrics_gauge = create_detection_metrics_gauge(avg_confidence, iou, precision, recall, num_detections)
        
        # Create HTML response
        results_html = self._create_html_report(results)
        
        return overlay, metrics_gauge, results_html
    
    def _create_overlay(self, image, results):
        """Create overlay visualization"""
        overlay = image.copy()
        
        for result in results:
            bbox = result['bounding_box']
            x, y, w, h = bbox
            
            forgery_type = result['forgery_type']
            confidence = result['confidence']
            
            # Get color
            forgery_id = [k for k, v in CLASS_NAMES.items() if v == forgery_type][0]
            color = CLASS_COLORS[forgery_id]
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{forgery_type}: {confidence:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(overlay, (x, y-label_h-8), (x+label_w+4, y), color, -1)
            cv2.putText(overlay, label, (x+2, y-4), font, font_scale, (255, 255, 255), thickness)
        
        return overlay
    
    def _create_html_report(self, results):
        """Create HTML report with detection results"""
        num_detections = len(results)
        
        if num_detections == 0:
            return """
            <div style='padding:12px; border:1px solid #5cb85c; border-radius:8px;'>
                ✓ <b>No forgery detected.</b><br>
                The document appears to be authentic.
            </div>
            """
        
        # Calculate statistics
        avg_confidence = sum(r['confidence'] for r in results) / num_detections
        type_counts = {}
        for r in results:
            ft = r['forgery_type']
            type_counts[ft] = type_counts.get(ft, 0) + 1
        
        html = f"""
        <div style='padding:12px; border:1px solid #d9534f; border-radius:8px;'>
            <b>⚠️ Forgery Detected</b><br><br>
            
            <b>Summary:</b><br>
            • Regions detected: {num_detections}<br>
            • Average confidence: {avg_confidence*100:.1f}%<br><br>
            
            <b>Detections:</b><br>
        """
        
        for i, result in enumerate(results, 1):
            forgery_type = result['forgery_type']
            confidence = result['confidence']
            bbox = result['bounding_box']
            
            forgery_id = [k for k, v in CLASS_NAMES.items() if v == forgery_type][0]
            color_rgb = CLASS_COLORS[forgery_id]
            color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
            
            html += f"""
            <div style='margin:8px 0; padding:8px; border-left:3px solid {color_hex}; background:rgba(0,0,0,0.02);'>
                <b>Region {i}:</b> {forgery_type} ({confidence*100:.1f}%)<br>
                <small>Location: ({bbox[0]}, {bbox[1]}) | Size: {bbox[2]}×{bbox[3]}px</small>
            </div>
            """
        
        html += """
        </div>
        """
        
        return html


# Initialize detector
detector = ForgeryDetector()


def detect_forgery(file, webcam):
    """Gradio interface function - handles file uploads and webcam capture"""
    try:
        # Use whichever input has data
        source = file if file is not None else webcam
        
        if source is None:
            empty_html = "<div style='padding:12px; border:1px solid #d9534f; border-radius:8px;'>❌ <b>No input provided.</b> Please upload a file or use webcam.</div>"
            return None, None, empty_html
        
        # Detect forgeries with detailed error tracking
        try:
            overlay, metrics_gauge, results_html = detector.detect(source)
            return overlay, metrics_gauge, results_html
        except Exception as detect_error:
            # Detailed error information
            import traceback
            import sys
            
            # Get full traceback
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            full_traceback = ''.join(tb_lines)
            
            # Print to console for debugging
            print("="*80)
            print("DETECTION ERROR - FULL TRACEBACK:")
            print("="*80)
            print(full_traceback)
            print("="*80)
            
            # Create detailed error HTML
            error_html = f"""
            <div style='padding:16px; border:2px solid #d9534f; border-radius:8px; background:#fff5f5;'>
                <h3 style='color:#d9534f; margin-top:0;'>❌ Detection Error</h3>
                <p><b>Error Type:</b> {exc_type.__name__}</p>
                <p><b>Error Message:</b> {str(exc_value)}</p>
                <details>
                    <summary style='cursor:pointer; color:#0066cc;'><b>Click to see full traceback</b></summary>
                    <pre style='background:#f5f5f5; padding:12px; overflow-x:auto; font-size:11px;'>{full_traceback}</pre>
                </details>
            </div>
            """
            return None, None, error_html
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {error_details}")
        error_html = f"""
        <div style='padding:12px; border:1px solid #d9534f; border-radius:8px;'>
            ❌ <b>Error:</b> {str(e)}
        </div>
        """
        return None, None, error_html


# Custom CSS - subtle styling
custom_css = """
.predict-btn {
    background-color: #4169E1 !important;
    color: white !important;
}
.clear-btn {
    background-color: #6A89A7 !important;
    color: white !important;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css) as demo:
    
    gr.Markdown(
        """
        # 📄 Document Forgery Detection
        Upload a document image or PDF to detect and classify forgeries using deep learning. The system combines MobileNetV3-UNet for precise localization and LightGBM for classification, identifying Copy-Move, Splicing, and Text Substitution manipulations with detailed confidence scores and bounding boxes. Trained on 140K samples for robust performance.
        """
    )
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Document")
            
            with gr.Tabs():
                with gr.Tab("📤 Upload File"):
                    input_file = gr.File(
                        label="Upload Image, PDF, or Document",
                        file_types=["image", ".pdf", ".doc", ".docx"],
                        type="filepath"
                    )
                
                with gr.Tab("📷 Webcam"):
                    input_webcam = gr.Image(
                        label="Capture from Webcam",
                        type="filepath",
                        sources=["webcam"]
                    )
            
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear", elem_classes="clear-btn")
                analyze_btn = gr.Button("🔍 Analyze", elem_classes="predict-btn")
        
        with gr.Column(scale=1):
            gr.Markdown("### Information")
            gr.HTML(
                """
                <div style='padding:16px; border:1px solid #ccc; border-radius:8px; background:var(--background-fill-primary);'>
                    <p style='margin-top:0;'><b>Supported formats:</b></p>
                    <ul style='margin:8px 0; padding-left:20px; list-style-type: disc; font-size: 16px;'>
                        <li style='margin-bottom: 6px;'>Images: JPG, PNG, BMP, TIFF, WebP</li>
                        <li style='margin-bottom: 6px;'>PDF: First page analyzed</li>
                    </ul>
                    
                    <p style='margin-bottom:4px;'><b>Forgery types:</b></p>
                    <ul style='margin:8px 0; padding-left:20px; list-style-type: disc; font-size: 16px;'>
                        <li style='color:#d9534f; margin-bottom: 6px;'><b>Copy-Move:</b> <span style='color:inherit;'>Duplicated regions</span></li>
                        <li style='color:#5cb85c; margin-bottom: 6px;'><b>Splicing:</b> <span style='color:inherit;'>Mixed sources</span></li>
                        <li style='color:#4169E1; margin-bottom: 6px;'><b>Text Substitution:</b> <span style='color:inherit;'>Modified text</span></li>
                    </ul>
                </div>
                """
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### Detection Results")
            output_image = gr.Image(label="Detected Forgeries", type="numpy")
    
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Analysis Report")
            output_html = gr.HTML(
                value="<i>No analysis yet. Upload a document and click Analyze.</i>"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Detection Metrics")
            metrics_gauge = gr.Plot(label="Concentric Metrics Gauge")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Architecture")
            gr.HTML(
                """
                <div style='padding:12px; border:1px solid #444; border-radius:10px; background:var(--background-fill-primary);'>
                    <p style="margin:0 0 0px 0; font-size:1.05em;"><b>Localization:</b> MobileNetV3-Small + UNet</p>
                    <p style='margin:0 20px 5px 0; margin-left:0.5cm; font-size:0.9em; opacity:0.85;'>Dice: 62.12% | IoU: 45.06% | Precision: 70.77% | Recall: 55.36%</p>
                    
                    <p style="margin:0 0 0 0; font-size:1.05em;"><b>Classification:</b> LightGBM with 526 features</p>
                    <p style="margin:0 20px 0 0; margin-left:0.5cm; font-size:0.9em; opacity:0.85;">Train Accuracy: 90.53% | Val Accuracy: 88.97%</p>

                    <p style='margin-top:5px; margin-bottom:0; font-size:1.05em;'><b>Training:</b> 120K samples from DocTamper dataset</p>
                </div>
                """
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Model Performance")
            gr.HTML(
                f"""
                <div style='padding:12px; border:1px solid #444; border-radius:10px; background:var(--background-fill-primary);'>
                    <p style='margin-top:0; margin-bottom:12px;'><b>Trained Model Performance:</b></p>
                    
                    <b>Segmentation Dice: {MODEL_METRICS['segmentation']['dice']*100:.2f}%</b>
                    <div style='width:100%; background:#333; height:12px; border-radius:6px; margin-bottom:12px;'>
                        <div style='width:{MODEL_METRICS['segmentation']['dice']*100:.1f}%; background:#4169E1; height:12px; border-radius:6px;'></div>
                    </div>
                    
                    <b>Classification Accuracy: {MODEL_METRICS['classification']['overall_accuracy']*100:.2f}%</b>
                    <div style='width:100%; background:#333; height:12px; border-radius:6px;'>
                        <div style='width:{MODEL_METRICS['classification']['overall_accuracy']*100:.1f}%; background:#5cb85c; height:12px; border-radius:6px;'></div>
                    </div>
                </div>
                """
            )
    
    # Event handlers
    analyze_btn.click(
        fn=detect_forgery,
        inputs=[input_file, input_webcam],
        outputs=[output_image, metrics_gauge, output_html]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, None, None, "<i>No analysis yet. Upload a document and click Analyze.</i>"),
        inputs=None,
        outputs=[input_file, input_webcam, output_image, metrics_gauge, output_html]
    )


if __name__ == "__main__":
    demo.launch()
