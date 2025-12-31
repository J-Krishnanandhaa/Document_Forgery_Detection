# Hybrid Document Forgery Detection & Localization System

A robust, hybrid (Deep Learning + Classical Machine Learning) system for multi-type document forgery detection and localization, designed to generalize across documents, receipts, and natural images, while remaining lightweight and deployable on CPU / edge devices.

---

## Problem Statement

Document forgeries such as copy-move, splicing, and text substitution are increasingly realistic and diverse. Pure deep-learning solutions often overfit to specific datasets, while classical forensic methods lack robustness.
This project addresses the problem using a strict hybrid architecture:

• Deep learning for pixel-level localization  
• Classical ML for interpretable forgery classification  

---

## Objectives

• Unified multi-type forgery detection and localization  
• Robustness to real-world degradation (scan noise, blur, compression)  
• Lightweight and real-time inference  
• Generalization across diverse document types  

---

## Final System Pipeline

Input Image / PDF  
        ↓  
Dataset-Aware Preprocessing  
        ↓  
Deep Forgery Localization  
(MobileNetV3-Small + UNet-Lite)  
        ↓  
Binary Forgery Mask  
        ↓  
Morphological Refinement  
        ↓  
Region Extraction  
        ↓  
Hybrid Feature Extraction  
        ↓  
LightGBM Classification  
        ↓  
Post-Processing  
        ↓  
Forgery Mask + JSON + Visualization  

---

## Supported Inputs

• Formats: JPG, PNG, PDF  
• Preferred Resolution: ≥ 300 DPI  
• Sources:  
o Scanned documents  
o Camera-captured documents  

---

## Preprocessing (Deterministic)

Applied to all datasets:

1. Convert to RGB  
2. Deskew document  
3. Resize to 384 × 384  
4. Normalize pixels to [0,1]  
5. Estimate noise using:  
   o Laplacian variance  
   o Wavelet-based noise estimation  
6. Apply conditional denoising only if noise exceeds threshold:  
   o Median filter (3×3)  
   o Gaussian filter (σ ≤ 0.8)  

This preserves subtle tampering artifacts.

---

## Training-Only Data Augmentation (Dataset-Aware)

• Noise addition  
• Motion blur  
• JPEG compression  
• Lighting variation  
• Perspective distortion  
• Stain & fold simulation (receipts)  

---

## Deep Localization Network

### Encoder

MobileNetV3-Small (ImageNet pretrained)

Chosen for:

• Stroke-level and texture preservation  
• Robustness to compression and blur  
• Edge and CPU deployment efficiency  

### Decoder

UNet-Lite

• Skip connections from encoder stages  
• Bilinear upsampling  
• Depthwise separable convolutions  

### Output

• Single-channel forgery probability mask  

### Loss Function

Loss = BCEWithLogits + Dice Loss  

---

## Mask Refinement

Post-inference processing:

• Morphological closing (fill broken strokes)  
• Morphological opening (remove noise)  
• Remove very small regions (< 0.1% of image area)  

---

## Region Extraction

• Connected component analysis (8-connectivity)  
• For each region:  
o Bounding box  
o Binary mask  
o Cropped region image  

This bridges deep learning and classical ML.

---

## Hybrid Feature Extraction (Per Region)

### 1. Deep Features

• Extracted from decoder feature maps  
• Global Average Pooling  

### 2. Statistical & Shape Features

• Area, perimeter  
• Aspect ratio  
• Solidity, eccentricity  
• Entropy  

### 3. Frequency-Domain Features

• DCT coefficients  
• High-frequency energy  
• Wavelet sub-band energy  

### 4. Noise & ELA Features

• Error Level Analysis (mean, variance)  
• Noise residual variance  

### 5. OCR-Consistency Features (Text Documents)

• OCR confidence deviation  
• Character spacing irregularity  
• Stroke width variation  

### Feature Fusion

• Concatenation  
• StandardScaler normalization  
• Missing-value handling enabled  

---

## Forgery Classification

### Classifier

LightGBM (Multiclass, Region-Wise)

### Target Forgery Types

copy_move  
splicing  
text_substitution  

### Why LightGBM

• Handles heterogeneous numerical features  
• Robust to small & imbalanced datasets  
• Interpretable (feature importance, SHAP)  
• Strong cross-dataset generalization  

### Inference Rule

• Region discarded if confidence < 0.60  

---

## False Positive Removal

• Confidence thresholding  
• Region size filtering  
• Deep mask consistency check  

---

## Outputs

### Visual

• Final forgery mask  
• Overlay visualization  

### JSON Output

{
  "region_id": 1,
  "bounding_box": [x, y, w, h],
  "forgery_type": "text_substitution",
  "confidence": 0.86,
  "mask_probability_mean": 0.72
}

---

## Training Strategy

• Single shared model  
• Interleaved batch mixing  
• Leave-One-Dataset-Out validation  

### Metrics

• IoU  
• Dice  
• Precision  
• Recall  
• F1-Score  

---

## Deployment

• ONNX export  
• Optional quantization  
• CPU and edge device support  

---

## Dataset-Specific Instructions

### DocTamper

• Load images & masks via LMDB  
• Use pixel-level masks directly  
• Enable JPEG & noise augmentation  
• Skip denoising unless required  

✔ Fully supported  

---

### RTM (Real Text Manipulation)

• Load from JPEGImages/ and SegmentationClass/  
• Use provided train/test splits  
• Enable motion blur & lighting augmentation  
• Preserve thin strokes carefully  

✔ Fully supported  

---

### CASIA v1.0

• Image-level labels only  
• Treat entire image as a single region  
• Use for classification and feature learning  
• Localization is weakly supervised  

⚠ Partial (classification-focused)  

---

### Find-It-Again (Receipts)

• Use provided train/val/test splits  
• Convert bounding boxes to region masks (optional)  
• Enable OCR features  
• Enable perspective & stain augmentation  

✔ Fully supported

