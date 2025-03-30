# Medicinal Plant Leaf Identification

## Overview
This project aims to classify medicinal plant leaves using a combination of **Convolutional Neural Networks (CNN)** and **Gray-Level Co-occurrence Matrix (GLCM) features** with a **Random Forest Classifier**. The ensemble approach leverages both deep learning and traditional machine learning to improve classification accuracy.

## Dataset

**Indian Medicinal Leaves Image Datasets**
- **Published:** 5 May 2023
- **Version:** 3
- **DOI:** [10.17632/748f8jkphb.3](https://data.mendeley.com/datasets/748f8jkphb/3)
- **Contributors:** Pushpa B R, Shobha Rani
- **Description:** A collection of medicinal plant images captured under varying backgrounds and lighting conditions.
- **License:** CC BY 4.0

## Features
- **CNN Model:** EfficientNetB0-based deep learning classifier.
- **GLCM Feature Extraction:** Extracts texture-based features for traditional classification.
- **Random Forest Classifier:** Trained on extracted GLCM features.
- **Ensemble Learning:** A weighted combination (70% CNN, 30% RF) for final classification.
- **Gradio Interface:** Interactive web-based prediction system.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/medicinal-leaf-id.git
   cd medicinal-leaf-id
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # For Linux/macOS
   .venv\Scripts\activate  # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure that the dataset is available in the correct directory:
   ```
   D:\MED_LEAF_ID-1\dataset\Medicinal Leaf dataset
   ```
5. Generate class labels if not available:
   ```bash
   python generate_class_labels.py
   ```

## Model Performance
### **CNN Model (EfficientNetB0)**
- **Accuracy:** 92.4%
- **Loss:** 0.23
- **Precision:** 91.8%
- **Recall:** 92.1%
- **F1-score:** 92.0%

### **Random Forest Classifier (GLCM Features)**
- **Accuracy:** 87.6%
- **Precision:** 86.9%
- **Recall:** 87.2%
- **F1-score:** 87.0%

### **Ensemble Model (70% CNN, 30% RF)**
- **Accuracy:** 94.1%
- **Precision:** 93.7%
- **Recall:** 93.9%
- **F1-score:** 93.8%

## Directory Structure
```
MED_LEAF_ID/
│── dataset/
│   ├── Medicinal Leaf dataset/
│── data/
│   ├── glcm_features.csv
│── models/
│   ├── plant_classifier.pkl
│   ├── efficientnetb0_leaf_model.pth
│── scripts/
│   ├── train_rf.py
│   ├── train_cnn.py
│   ├── train_ensemble.py
│── app.py  # Gradio Interface
│── README.md
│── requirements.txt
```

## Contact
For any inquiries, reach out to **Aadhithya Prakash** at **aadhithyasubha2018@gmail.com**.

## Citation
If you use this dataset or project, kindly cite:
```
B R, Pushpa; Rani, Shobha (2023), “Indian Medicinal Leaves Image Datasets”, Mendeley Data, V3, doi: 10.17632/748f8jkphb.3
```

## Future Work
- Improve CNN model performance using fine-tuning and hyperparameter optimization.
- Explore advanced ensemble techniques for better decision fusion.
- Develop a mobile-friendly application for real-world plant identification.

If you found this project useful, give it a **star** on GitHub!

