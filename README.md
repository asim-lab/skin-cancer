# Hybrid Model with Attention Mechanism for Melanoma Detection

Hybrid Model for Image Classification with Feature Visualization
This project presents a hybrid deep learning approach for binary image classification, leveraging ResNet-50 and VGG-16 feature extractors with attention mechanisms. It integrates advanced data augmentation, feature visualization, and t-SNE analysis to enhance classification accuracy and model interpretability.

## Problem Statement

Traditional image classification models often struggle with feature extraction, leading to misclassification and poor generalization. This project addresses these challenges by fusing deep features from ResNet-50 and VGG-16 while incorporating attention mechanisms to refine the feature representation. The goal is to create a robust and explainable AI model for reliable classification.

## Dataset
The PH2 dataset can be accessed at https://www.kaggle.com/datasets/athina123/ph2dataset 
and ISIC 2016-2017 dataset is available at https://challenge.isic-archive.com/data/

## Methodology

### 1. Data set and pre-processing
The dataset is divided into training and test sets with labelled images.
Preprocessing includes random cropping, flipping, rotating, affine transformations and normalization to improve model generalization.

### 2. Hybrid deep learning model
Feature fusion combines deep features from ResNet-50 and VGG-16 to capture global and local image representations.
The attention mechanism uses channel and spatial attention modules to highlight critical image features.
Transfer learning fine-tunes the pre-trained ResNet-50 and VGG-16 backbones for improved performance.

### 3. Feature visualization and analysis
Feature map visualization displays intermediate representations of features to analyze what the model is learning.
t-SNE projection reduces the high-dimensional feature space to 2D to better interpret class separation.

### 4. Model training and evaluation
The k-fold cross-validation ensures robustness.
Performance metrics include accuracy, F1-score, ROC-AUC and confusion matrices.
Optimization is performed using the Adam optimizer, learning rate planning and dropout to prevent overfitting.

### Results
The hybrid model demonstrated high accuracy and interpretability, outperforming standard CNNs. Feature visualization techniques confirmed that the model effectively identifies discriminative features and improves classification reliability.
