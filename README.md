# Multi-Hilbert-Remote-Sensing-Scene-Classification

This project implements an advanced remote sensing scene classification system that combines Convolutional Neural Networks (CNNs) with Hilbert curve-based feature extraction for improved accuracy.

## Overview

The system uses multi-scale Hilbert curve transformations to extract spatial features from remote sensing imagery, which are then fused with deep features from pre-trained CNNs to enhance classification performance. The approach is particularly effective for the UCMerced Land Use dataset.

## Features

- **Multi-scale Hilbert curve feature extraction**: Captures spatial patterns at different scales (64x64, 32x32, 16x16)
- **Enhanced statistical features**: Extracts mean, standard deviation, median, quartiles, entropy, and FFT coefficients
- **Model fusion architecture**: Combines CNN features with Hilbert curve features using attention mechanisms
- **Multiple backbone options**: Supports MobileNetV2, ResNet50, and VGG16 as CNN backbones
- **Fine-tuning capabilities**: Includes a two-phase training approach with fine-tuning

## Requirements

- TensorFlow 2.x
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- hilbertcurve

## Dataset

The code is designed to work with the UCMerced Land Use dataset, which contains 21 land-use classes with 100 images per class at a resolution of 256x256 pixels.
Dataset can be download from http://weegee.vision.ucmerced.edu/datasets/landuse.html or https://hyper.ai/cn/datasets/5431.

## Usage

1. Download the UCMerced Land Use dataset and place it in the `UCMerced_LandUse/Images` directory
2. Run the main script:

```
python Hilbert_Scene_Classification.py
```

## Model Architecture

The system implements two main models:
1. **Baseline Model**: A standard CNN-based classifier using pre-trained networks
2. **Fusion Model**: An enhanced model that combines CNN features with multi-scale Hilbert curve features

The fusion model architecture includes:
- CNN branch for extracting deep visual features
- Multiple Hilbert curve branches with scale-specific processing
- Attention mechanisms at both local and global levels
- Dense layers for feature fusion and classification

## Visualization

The code generates several visualizations:
- Model architecture diagrams
- Training and validation accuracy/loss curves
- Confusion matrices
- Classification reports

## Performance

The fusion model typically outperforms the baseline CNN model, demonstrating the value of incorporating Hilbert curve spatial features for remote sensing scene classification.
