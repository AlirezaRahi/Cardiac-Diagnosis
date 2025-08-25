# Fusion-Based Deep Learning Ensemble on MIT-BIH and PTB-XL ECG Databases for Enhanced Cardiac Diagnosis

This repository contains the implementation of our research on ECG-based cardiac diagnosis using ensemble deep learning approaches, as presented in our publication.

**Publication**: [Fusion-Based Deep Learning Ensemble on MIT-BIH and PTB-XL ECG Databases for Enhanced Cardiac Diagnosis](https://www.medrxiv.org/content/10.1101/2025.08.20.25334079v1)  
**DOI**: https://doi.org/10.1101/2025.08.20.25334079  
**Publication Date**: August 20, 2025

##  Abstract

Electrocardiogram (ECG) analysis plays a critical role in the early detection and diagnosis of cardiac abnormalities. In this study, we propose a fusion-based deep learning ensemble framework that integrates two well-established public ECG databases, MIT-BIH Arrhythmia Database and PTB-XL, to develop a robust and automated cardiac diagnostic system. Our framework employs two base deep learning models, a CNN+LSTM hybrid and a DenseNet1D-inspired network, and combines their predictive features through a meta-learner based on Gradient Boosting. This multi-model integration, designed as a mini doctor for the heart, leverages the complementary strengths of both datasets and models. Experimental results demonstrate that the ensemble achieves near-perfect performance with Accuracy up to 100% and ROC-AUC of 1.000, surpassing the performance of individual models. These findings highlight the potential of database fusion and model ensembling for building reliable and scalable solutions in computer-aided cardiac diagnosis.

## 🏆 Key Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| CNN+LSTM | 97.8% | 0.995 |
| DenseNet1D | 98.1% | 0.997 |
| Gradient Boosting Ensemble | **100%** | **1.000** |

##  Project Structure
project-root/
│
├── models/
│ ├── cnn_lstm_best.keras # Best CNN+LSTM model
│ ├── cnn_lstm_final.keras # Final CNN+LSTM model
│ ├── densenet1d_best.keras # Best DenseNet1D model
│ ├── densenet1d_final.keras # Final DenseNet1D model
│ └── meta_learner_ecg.pkl # Gradient Boosting meta-learner
│
├── datasets/
│ └── ecg_dataset/
│ ├── ecg_data.npy # ECG signal data
│ └── ecg_labels.npy # Corresponding labels
│
├── ECG_Ensemble.py # Main implementation code
└── README.md # This file



##  Model Architectures

### 1. CNN + LSTM Hybrid
- **Architecture**: Conv1D(64) → MaxPool → Conv1D(128) → MaxPool → Conv1D(256) → MaxPool → LSTM(128) → Dense(5)
- **Input**: Raw ECG signals (2500 samples)
- **Output**: 5-class classification probabilities

### 2. DenseNet1D-inspired Network
- **Architecture**: Residual blocks with skip connections
- **Features**: Batch normalization, residual connections, global average pooling
- **Output**: 5-class classification probabilities

### 3. Gradient Boosting Meta-Learner
- **Algorithm**: Gradient Boosting Classifier
- **Input**: Concatenated features from both base models
- **Parameters**: n_estimators=200, learning_rate=0.05, max_depth=3

## 📊 Dataset

The integrated dataset combines:
- **MIT-BIH Arrhythmia Database**: 48 half-hour ECG recordings
- **PTB-XL ECG Database**: 21,801 clinical 12-lead ECG recordings
- **Total Samples**: 7,023 ECG signals across 5 classes
- **Signal Length**: 2500 samples per ECG recording
- **Classes**: Normal and 4 types of cardiac abnormalities

## 🚀 Installation & Requirements

```bash
pip install tensorflow keras scikit-learn matplotlib numpy joblib
💻 Usage
python
python ECG_Ensemble.py

🔐 Model Availability
Due to the sensitive nature of the trained models and intellectual property protection, the actual trained model files are not publicly hosted in this repository.

The complete source code for training, evaluation, and ensemble implementation is provided, allowing researchers to replicate our results exactly.

If you require access to the pre-trained models for academic collaboration or research verification, please contact me directly.

 Author
Alireza Rahi

 Email: alireza.rahi@outlook.com

 LinkedIn: https://www.linkedin.com/in/alireza-rahi-6938b4154/

 GitHub: https://github.com/AlirezaRahi

 License
All Rights Reserved.

Copyright (c) 2025 Alireza Rahi

Unauthorized access, use, modification, or distribution of this software is strictly prohibited without explicit written permission from the copyright holder.