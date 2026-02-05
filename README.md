# DeepFake Detection System
Classical ML + Fine-Tuned ResNet50 | Video-Level Inference | Streamlit Deployment

---

## Try Model Here

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepfake-video-detection.streamlit.app/)

---


## Overview
This project presents an end-to-end deepfake detection system focused on identifying face-swapped deepfake videos.
The framework combines large-scale face-level preprocessing, classical machine learning baselines, and a two-stage fine-tuned ResNet50 deep learning model, followed by video-level aggregation and real-time deployment on Streamlit.

## What This Project Does
- Detects **face-swapped deepfake videos**
- Works at **frame level → aggregates to video level**
- Compares **classical ML vs deep learning**
- Optimized for **recall (don’t miss deepfakes)**
- Deployed as a **Streamlit web app (CPU inference)**

## Features
- Built a custom dataset (**~33,000 aligned face images**)
- Prevented data leakage using **video-wise splitting**
- Hybrid approach: **HOG / PCA / Deep features + ML models**
- **Two-stage fine-tuned ResNet50**
- **Video-level aggregation** (majority / mean voting) for **better performace** and **eliminating dependency on single frame**
- **Threshold tuning**
- **Live on** Stream lit 

## Faces
![Project Demo](outputs/eda/sample_grid.png)

## Tech Stack
- **Language:** Python
- **DL Framework:** TensorFlow
- **ML:** scikit-learn
- **Vision:** OpenCV, MTCNN
- **Deployment:** Streamlit Cloud

## Pipeline
![Project Demo](outputs/A1.png)


Models Used
---
### Classical Models (BaseLine)
- Logistic Regression
- Naive Bayes
- Random Forest

### Trained on :
- HOG Features
- PCA embeddings
- ResNet50 deep features

---

### Deep Learning
- **ResNet50** (ImageNet pretrained)
- Two-stage fine-tuning
- Weighted Binary Cross-Entropy
- Adam optimizer

---

## Results 
### Frame and Video level Accuracy
![Project Demo](outputs/A2.png) 

---

### ROC AUC
![Project Demo](outputs/A3.png)

---

### Model Evaluation
![Project Demo](outputs/A4.png)

---
## Deployment
- GPU used for training, **CPU only inference**
- Upload video → get real/fake verdict

## Limitations
- Single manipulation type (**DeepFakes**)
- Face detector can fail in extreme cases

## Official Report
[Detailed Report](INT_395_Report_Aryan.pdf)

