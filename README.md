# 🗑️ Automatic Garbage Classification using CNN-ELM Ensemble

This project presents a high-performance deep learning solution for automatic garbage classification using a **hybrid ensemble of Convolutional Neural Networks (CNNs)** and **Extreme Learning Machines (ELMs)**. The system classifies waste images into six categories with impressive accuracy and minimal training time.

---

## 🚀 Overview

Garbage classification is critical for sustainable waste management. Traditional methods like manual sorting are labor-intensive and error-prone. This project proposes an **automated image-based classification system** that combines the powerful feature extraction capability of CNNs with the fast training nature of ELMs.

A **voting-based ensemble model** was implemented using five CNN architectures:
- ResNet50
- InceptionV3
- DenseNet121
- VGG19
- EfficientNetB0

Each CNN model extracts image features, which are then classified using an ELM. Predictions are combined via **hard voting** to generate final class predictions.

---

## 🧠 Key Features

- ✅ **CNN-ELM Hybrid Model** – Combines deep feature extraction with fast classification
- ✅ **Voting Ensemble** – Improves prediction robustness
- ✅ **High Accuracy** – Achieved 94.79% accuracy on the TrashNet dataset (non-augmented)
- ✅ **Low Training Time** – Trains in a single epoch thanks to ELM
- ✅ **Modular Architecture** – Easily replace or fine-tune CNNs

---

## 🧾 Dataset

- **Source**: [TrashNet – Garbage Classification on Kaggle]
- **Classes**:
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash
- **Total Images**: 2527

The dataset was split into:
- Training: 80%
- Validation: 10%
- Testing: 10%

---

## 📊 Model Performance

| Model                | Accuracy (No Aug.) | Accuracy (With Aug.) |
|---------------------|--------------------|-----------------------|
| ResNet50-ELM        | 57.64%             | 32.29%                |
| InceptionV3-ELM     | 93.75%             | 72.92%                |
| VGG19-ELM           | 85.42%             | 65.62%                |
| DenseNet121-ELM     | 93.75%             | 79.51%                |
| EfficientNetB0-ELM  | 79.51%             | 45.14%                |
| **Ensemble (CNN-ELM)** | **94.79%**         | **76.04%**             |

---

## 🛠️ How It Works

1. **Preprocessing**:
   - Resize images to 256x256
   - Normalize pixel values
   - Split dataset (train/val/test)
   - Optional: Data augmentation (flip, rotate)

2. **Feature Extraction**:
   - Use pre-trained CNNs (ImageNet weights) without top classification layers
   - Apply average pooling to get feature vectors

3. **Classification with ELM**:
   - Use `hpelm` for fast training of ELM classifiers
   - Tune hidden neurons per model using validation set

4. **Ensemble Voting**:
   - Aggregate predictions from all five CNN-ELMs
   - Final class is selected via majority vote

---

## 🔧 Installation

Install the required Python libraries:

```bash
pip install tensorflow scikit-learn hpelm matplotlib numpy pillow
