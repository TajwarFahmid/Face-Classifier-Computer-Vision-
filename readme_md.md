# Fine-Grained Facial Recognition & Transfer Learning Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end computer vision pipeline developed in TensorFlow/Keras evaluating baseline Convolutional Neural Networks (CNN) versus deep transfer learning (VGG16) for fine-grained multi-class portrait classification under low-sample constraints (~80–180 images per identity across 17 classes).

---

## 📌 Project Overview

Fine-grained visual categorization (FGVC) across human faces presents significant modeling challenges, including high intra-class variance (lighting, age, facial orientation) and low inter-class variance. 

This repository documents an empirical investigation comparing:
1. **Custom 3-Stage CNN Baseline:** Trained from scratch using Batch Normalization, Max Pooling, Dropout, and custom portrait-oriented data augmentation.
2. **Transfer Learning via ImageNet-Pretrained VGG16:** Feature extractor utilizing frozen convolutional blocks integrated with a custom fully-connected multi-layer perceptron (MLP) head.

### Key Results
* **Custom Baseline CNN:** Reached **~45.1%** top-1 validation accuracy across 17 classes. It exhibited severe sample inefficiency, gradient instability at deeper epochs, and susceptibility to majority-class prediction collapse when trained beyond 50 epochs.
* **VGG16 Transfer Learning:** Lifted top-1 validation accuracy to **~66.6%** (+21.5% absolute improvement) with stable loss convergence, demonstrating the necessity of high-level inductive visual priors when dataset cardinality is constrained.

---

## 🏗️ Model Architectures

### 1. Custom Baseline CNN
* **Input Resolution:** $150 \times 150 \times 3$
* **Feature Extraction:**
  * Block 1: `Conv2D(32, 3x3, ReLU)` $\rightarrow$ `MaxPooling2D(2x2)` $\rightarrow$ `BatchNormalization`
  * Block 2: `Conv2D(64, 3x3, ReLU)` $\rightarrow$ `MaxPooling2D(2x2)` $\rightarrow$ `BatchNormalization`
  * Block 3: `Conv2D(128, 3x3, ReLU)` $\rightarrow$ `MaxPooling2D(2x2)`
* **Classification Head:**
  * `Flatten` $\rightarrow$ `Dense(512, ReLU)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dense(17, Softmax)`
* **Optimization:** Adam ($\alpha = 10^{-3}$) with custom step-decay schedule.

### 2. VGG16 Transfer Learning Pipeline
* **Base Backbone:** VGG16 (ImageNet weights, top classification layers truncated, all base layers frozen `trainable = False`).
* **Custom Classification Head:**
  * `Flatten`
  * `Dense(512, ReLU)`
  * `Dropout(0.5)`
  * `BatchNormalization`
  * `Dense(17, Softmax)`
* **Optimization:** Adam ($\alpha = 10^{-4}$ reduced for transfer stability) with exponential step-decay schedule.

---

## 📊 Dataset Structure & Preprocessing

The dataset comprises **1,470 training images** and **350 validation images** categorized into 17 distinct celebrity identities.

```
data/
├── train/
│   ├── Angelina Jolie/
│   ├── Brad Pitt/
│   ├── Denzel Washington/
│   ├── Johnny Depp/
│   ├── Robert Downey Jr/
│   └── ... (17 classes)
└── val/
    ├── Angelina Jolie/
    └── ...
```

### Domain-Calibrated Data Augmentation
To simulate real-world framing variation without inducing catastrophic identity distortion, custom data augmentation parameters were applied via `ImageDataGenerator`:
* **Rescaling:** Pixel intensity normalization to $[0, 1]$
* **Rotation:** Minimal range ($\pm 5^\circ$) to preserve eye-line geometry
* **Spatial Shift:** Width/Height shifts bounded at $5\%$
* **Brightness Adjustment:** Random illumination scaling between $[0.9, 1.1]$
* **Zoom Range:** Subtly cropped between $[0.9, 1.1]$
* **Horizontal Flip:** Activated for lateral facial invariance

---

## 📈 Training Dynamics & Results

| Model Architecture | Parameters | Epochs | Top-1 Val Accuracy | Final Val Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Custom 3-Stage CNN** | ~2.1M | 50 | 45.14% | 1.596 |
| **VGG16 Transfer Learning** | ~15.8M (frozen base) | 30 | **66.57%** | **1.085** |

### Qualitative Inference Validation
Model performance was validated on unseen wild portrait test images, correctly identifying complex poses and illumination:
* ✅ Robert Downey Jr. (`True Positive`)
* ✅ Johnny Depp (`True Positive`)
* ✅ Nicole Kidman (`True Positive`)
* ✅ Scarlett Johansson (`True Positive`)
* ✅ Will Smith (`True Positive`)
* ⚠️ Tom Cruise (`Misclassified as Hugh Jackman` due to similar jawline morphology and facial angle under high crop zoom)

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/face-classification-transfer-learning.git
cd face-classification-transfer-learning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Training
Run the Jupyter/Colab notebook:
```bash
jupyter notebook Computer_Vision_Updated_project_.ipynb
```

### 4. Single Image Inference
```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('best_model.h5')

# Preprocess image
img_path = 'path/to/test_image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

# Predict
predictions = model.predict(img_tensor)
predicted_idx = np.argmax(predictions[0])
print(f"Predicted Class Index: {predicted_idx}")
```

---

## 🔬 Key Engineering Insights & Future Work

1. **Catastrophic Forgetting & Overfitting:** Attempting to train deeper CNNs from scratch on small image volumes ($\sim 80$ per class) triggers extreme overfitting or gradient dispersion, where the network defaults to the statistical majority class.
2. **Transfer Learning Superiority:** Pretrained representations from ImageNet capture robust spatial hierarchies (Gabor-like edge detectors, textures, facial components) that transfer effectively even with frozen feature weights.
3. **Proposed Upgrades:**
   * **Face Alignment / Detection Frontend:** Integrate a lightweight face detector (e.g., MTCNN or YOLOv8-Face) prior to classification to strip extraneous background noise.
   * **Deep Metric Learning:** Replace standard categorical cross-entropy with Triplet Loss or ArcFace (Additive Angular Margin Loss) to optimize identity embeddings rather than classification probabilities.
   * **Modern Mobile Backbones:** Transition from VGG16 to MobileNetV3 or EfficientNetB0 for lower latency edge deployment on platforms like NVIDIA Jetson.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.