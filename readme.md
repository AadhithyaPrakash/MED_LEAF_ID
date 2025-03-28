# Medical Leaf Identification

## Overview

This project focuses on **Medical Leaf Identification** using a combination of **Gray-Level Co-occurrence Matrix (GLCM) features and Convolutional Neural Networks (CNNs)**. It aims to classify various medicinal leaves by leveraging both texture-based and deep learning approaches.

## Dataset

The dataset used for this project is the **Indian Medicinal Leaves Image Dataset**:

- **Published:** 5 May 2023  
- **Version:** 3  
- **DOI:** [10.17632/748f8jkphb.3](https://doi.org/10.17632/748f8jkphb.3)  
- **Contributors:** Pushpa B R, Shobha Rani  
- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/748f8jkphb/3)  
- **Description:** The dataset consists of images of Indian medicinal plants with varying backgrounds and without environmental constraints.

## Project Structure

```
MED_LEAF_ID/
│-- data/
│   ├── original_images/  # Raw images from the dataset
│   ├── augmented_images/  # Augmented dataset for better training
│   ├── glcm_features/  # Extracted GLCM features
│   ├── cnn_preprocessed/  # Preprocessed images for CNN training
│
│-- notebooks/
│   ├── EDA.ipynb  # Exploratory Data Analysis
│   ├── augmentation.ipynb  # Data Augmentation
│   ├── preprocessing_glcm.ipynb  # Preprocessing for GLCM feature extraction
│   ├── preprocessing_cnn.ipynb  # Image preprocessing for CNN
│   ├── training_glcm_model.ipynb  # GLCM-based classifier training
│   ├── training_cnn_model.ipynb  # CNN training
│
│-- models/
│   ├── glcm_model.pkl  # Trained GLCM classifier
│   ├── cnn_model.pth  # Trained CNN model
│
│-- scripts/
│   ├── glcm_extraction.py  # Script for extracting GLCM features
│   ├── cnn_preprocessing.py  # Image preprocessing script for CNN
│   ├── train_glcm.py  # Training script for GLCM model
│   ├── train_cnn.py  # Training script for CNN model
│
│-- requirements.txt  # Dependencies
│-- README.md  # Project documentation
│-- .gitignore  # Ignoring unnecessary files
```

## Methodology

This project employs a **hybrid approach** using both **GLCM and CNN-based classification**:

### 1. **GLCM-Based Classifier**

- Extracts texture features from grayscale images.
- Computes co-occurrence matrices and derives statistical features.
- Trains a machine learning model on extracted GLCM features.

### 2. **CNN-Based Classifier**

- Uses a deep learning model trained on the dataset.
- Includes image augmentation for better generalization.
- Implements a **Convolutional Neural Network (CNN)** for classification.

### 3. **Ensemble Approach (Future Work)**

- Combines both classifiers to improve accuracy.
- Merges texture and deep-learning features.

## Preprocessing Steps

### For **GLCM**

- Convert images to grayscale.
- Extract **contrast, correlation, energy, and homogeneity** features.
- Normalize extracted features for better classification.

### For **CNN**

- Resize images to **224×224**.
- Normalize pixel values.
- Augment images (rotation, flipping, brightness adjustment).
- Convert images into a format suitable for CNN training.

## Model Training

- **GLCM Model:** Trained using machine learning algorithms (SVM, Random Forest, etc.).
- **CNN Model:** Implemented using **PyTorch**, trained with **GPU acceleration**.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AadhithyaPrakash/MED_LEAF_ID.git
cd MED_LEAF_ID
```

### 2. Set up the environment

```bash
python -m venv base
source base/bin/activate  # On Windows: base\Scripts\activate
pip install -r requirements.txt
```

### 3. Run preprocessing and training

```bash
python scripts/glcm_extraction.py
python scripts/cnn_preprocessing.py
python scripts/train_glcm.py
python scripts/train_cnn.py
```

## Results

- **GLCM-based classifier** achieved an accuracy of **X%**.
- **CNN model** obtained an accuracy of **Y%** after fine-tuning.
- **Ensemble approach (Future work)** aims to boost accuracy further.

## Future Work

- Implement ensemble learning to merge GLCM and CNN features.
- Optimize hyperparameters for improved performance.
- Deploy the trained model as a web application.

## Contributors

- **Aadhithya Prakash** - Developer & Researcher
- Open for contributions! Feel free to fork and submit a PR.

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

Special thanks to the **Indian Medicinal Leaves Image Dataset** creators and the **open-source community** for providing resources for deep learning research.

---

### 🌿 **Medical Leaf Identification – Blending Machine Learning and Deep Learning for Precision Botany**
