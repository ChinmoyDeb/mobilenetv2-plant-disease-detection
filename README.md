# 🌿 MobileNetV2 Transfer Learning for Plant Disease Detection

A deep learning project implementing transfer learning on MobileNetV2 for automated plant disease classification with a focus on domain adaptation and real-world robustness.

This project reproduces and improves a two-phase transfer learning pipeline for Plant Disease Detection using the PlantDoc dataset.

---

# 📋 Table of Contents

* [Project Overview](#-project-overview)
* [Model Architecture](#️-model-architecture)
* [Dataset Information](#-dataset-information)
* [Methodology](#-methodology)
* [Training Pipeline](#-training-pipeline)
* [Results](#-results)
* [Performance Analysis](#-performance-analysis)
* [Training Visualizations](#-training-visualizations)
* [Experiment Tracking](#-experiment-tracking)
* [Technologies Used](#️-technologies-used)
* [Installation](#-installation)
* [Usage](#-usage)
* [Key Findings](#-key-findings)
* [Developer](#-developer)

---

# 🌱 Project Overview

This project tackles real-world plant disease detection using transfer learning and computer vision.

Starting from a pretrained MobileNetV2 model, a two-phase fine-tuning strategy was implemented to adapt the model from controlled laboratory images to noisy real-world agricultural field images.

### Key Objectives

* Improve model performance on out-of-distribution field data
* Implement efficient transfer learning strategies
* Maintain lightweight inference using MobileNetV2
* Achieve robust multi-class classification across 28 disease classes
* Improve domain adaptation from PlantVillage → PlantDoc

---

# 🏗️ Model Architecture

## Base Network

| Component          | Details                     |
| ------------------ | --------------------------- |
| Architecture       | MobileNetV2                 |
| Framework          | HuggingFace Transformers    |
| Input Resolution   | 224 × 224                   |
| Feature Extractor  | 16 inverted residual blocks |
| Final Feature Size | 1280                        |
| Output Classes     | 28                          |

---

## Custom Classification Head

```python
Sequential(
    Linear(1280 → 640)
    ReLU()
    Linear(640 → 320)
    Dropout(p=0.1)
    ReLU()
    Linear(320 → 28)
)
```

---

# 📊 Dataset Information

## Source Model Training — PlantVillage

| Feature     | Details               |
| ----------- | --------------------- |
| Images      | 10,878                |
| Classes     | 38                    |
| Environment | Controlled Laboratory |

---

## Transfer Learning — PlantDoc

| Split      | Images |
| ---------- | ------ |
| Train      | 1,873  |
| Validation | 469    |
| Test       | 236    |
| Total      | 2,578  |

### Domain Adaptation Challenge

PlantVillage contains clean laboratory images while PlantDoc contains:

* real-world field conditions
* varying lighting
* cluttered backgrounds
* blur and camera noise
* inconsistent image quality

---

# 🔬 Methodology

## Two-Phase Transfer Learning

### Phase 1 — Frozen Backbone Training

* Freeze MobileNetV2 backbone
* Train custom classifier head only
* Fixed BatchNorm layers in eval mode
* AdamW optimizer + warmup scheduler

### Result

* Validation Accuracy: 43.1%

---

### Phase 2 — Progressive Fine-Tuning

* Unfreeze last 3 MobileNetV2 blocks
* Keep BatchNorm frozen
* Fine-tune deeper feature representations
* Cosine Annealing Scheduler

### Result

* Best Validation Accuracy: 53.7%
* Final Test Accuracy: 45.76%

---

# 🔄 Training Pipeline

```text
PlantVillage Pretrained Model
            ↓
Freeze Backbone
            ↓
Train Custom Classifier
            ↓
Progressive Unfreezing
            ↓
Fine-Tune Deep Features
            ↓
Evaluate on PlantDoc
```

---

# 📈 Results

| Metric | Baseline | Final Model | Improvement |
|---|---|---|---|
| Validation Accuracy | 40.7% | 53.7% | +13.0% |
| Test Accuracy | 19.49% | 45.76% | +26.27% |
| Weighted F1 Score | 47.0% | 43.74% | -3.26% |

---

## Key Improvements Over Baseline

- Improved validation accuracy by **13%**
- Improved real-world PlantDoc test accuracy by **26.27%**
- Successfully adapted MobileNetV2 from laboratory images to noisy field images
- Improved domain generalization using progressive fine-tuning
- Achieved stronger feature adaptation using BatchNorm stabilization

---

# 📉 Performance Analysis

| Stage         | Accuracy |
| ------------- | -------- |
| Early Phase 1 | 12.4%    |
| Final Phase 1 | 43.1%    |
| Final Phase 2 | 53.7%    |

---

# 📸 Training Visualizations

## Complete Training History

![Training History](screenshots/training_history_plot.png)

---

## Validation Accuracy

![Validation Accuracy](screenshots/val_acc.png)

---

## Validation F1 Score

![Validation F1](screenshots/val_f1.png)

---

## Training Loss

![Training Loss](screenshots/train_loss.png)

---

## Validation Loss

![Validation Loss](screenshots/val_loss.png)

---

## Learning Rate Schedule

![Learning Rate](screenshots/learning_rate.png)

---


# 📊 Experiment Tracking

Tracked using Weights & Biases:

* train loss
* validation loss
* validation accuracy
* weighted F1 score
* learning rate
* epoch metrics
* batch loss

---

# ⚙️ Technologies Used

| Technology   | Purpose             |
| ------------ | ------------------- |
| Python       | Programming         |
| PyTorch      | Deep Learning       |
| Transformers | HuggingFace Models  |
| Torchvision  | Image Processing    |
| WandB        | Experiment Tracking |
| Scikit-learn | Metrics             |
| Google Colab | GPU Training        |

---

# 🚀 Installation

```bash
git clone https://github.com/ChinmoyDeb/mobilenetv2-plant-disease-detection.git
cd mobilenetv2-plant-disease-detection
```

```bash
pip install -r requirements.txt
```

---

# 💻 Usage

```python
from transformers import MobileNetV2ForImageClassification

model = MobileNetV2ForImageClassification.from_pretrained(
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)
```

---

# 🔍 Key Findings

* Progressive fine-tuning significantly improved feature adaptation
* BatchNorm freezing stabilized transfer learning
* Data augmentation improved generalization performance
* Domain adaptation from PlantVillage → PlantDoc was successful
* Real-world agricultural image classification remains highly challenging

---

# 👨‍💻 Developer

## Chinmoy Deb

### VIT Vellore

AI • Deep Learning • Computer Vision 🌱

---

# 📅 Last Updated

May 2026
