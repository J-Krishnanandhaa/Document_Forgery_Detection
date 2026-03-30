# Document Forgery Detection and Localization

## Overview
This project presents a **hybrid deep learning and gradient boosting framework** for detecting and localizing document forgeries. It combines segmentation-based localization with classification techniques to identify different types of document manipulation.

The system is designed to handle:
- Copy-Move Forgery
- Image Splicing
- Text Substitution

---

## Live Demo

Try the app here:
https://huggingface.co/spaces/DocForg/Document_Forgery_Detection

---

## Features
-  **Forgery Localization** using deep learning segmentation models  
-  **Forgery Classification** using Gradient Boosting (LightGBM/XGBoost)  
-  **Multi-class Detection** (Copy-Move, Splicing, Text Substitution)  
-  Efficient and scalable pipeline  
-  Strong performance across multiple metrics  

---

## Architecture
The proposed system follows a hybrid pipeline:

1. **Input Document Image**  
2. **Preprocessing**
   - Noise removal  
   - Resizing & normalization  
3. **Segmentation Model**
   - Detects forged regions  
4. **Feature Extraction**
   - Extracts region-based features  
5. **Classifier (Gradient Boosting)**
   - Classifies type of forgery  
6. **Output**
   - Forgery type + localized mask  

---

## Tech Stack
- **Python**  
- **TensorFlow / PyTorch** (for segmentation)  
- **LightGBM / XGBoost** (for classification)  
- **OpenCV** (image processing)  
- **NumPy & Pandas**  

---

## Dataset
DocTamper Dataset: https://github.com/qcf-568/DocTamper


---

## Installation

    git clone https://github.com/J-Krishnanandhaa/Document_Forgery_Detection
    cd document-forgery-detection
    pip install -r requirements.txt

---

## Results

| Metric Category | Metric | Score |
|----------------|--------|-------|
| Segmentation | Dice Coefficient | 0.6212 |
| Classification | Copy-Move Accuracy | 0.921 |
| Classification | Splicing Accuracy | 0.853 |
| Classification | Text Substitution Accuracy | 0.902 |
| Overall | Accuracy | 0.897 |

---

## Project Structure

    ├── data/
    ├── models/
    ├── src/
    │   ├── preprocessing/
    │   ├── segmentation/
    │   ├── feature_extraction/
    │   ├── classification/
    ├── train.py
    ├── predict.py
    ├── requirements.txt
    └── README.md

---

## How to Run the Project (Online)

The model is available as a live web app.

Open the demo here:  
https://huggingface.co/spaces/DocForg/Document_Forgery_Detection
```
1.
2.
3.
```


