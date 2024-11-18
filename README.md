# YOLOv1 Implementation in PyTorch

This repository contains an implementation of the YOLOv1 object detection model using PyTorch. The model is trained on the Pascal VOC 2007 dataset and utilizes a pre-trained ResNet-50 backbone for feature extraction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Plotting Test Images](#plotting-test-images)
  - [Plotting Loss Curves](#plotting-loss-curves)
- [Code Explanation](#code-explanation)
  - [Model Architecture](#model-architecture)
  - [Custom Loss Function](#custom-loss-function)
  - [Dataset Class](#dataset-class)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

YOLO (You Only Look Once) is a real-time object detection system that applies a single neural network to the full image. The network divides the image into regions and predicts bounding boxes and probabilities for each region.

This implementation leverages a pre-trained ResNet-50 model as the backbone and adds custom detection layers to perform object detection on the Pascal VOC dataset.

## Features

- **Pre-trained Backbone**: Uses ResNet-50 pre-trained on ImageNet for feature extraction.
- **Custom Detection Layers**: Adds convolutional and fully connected layers for bounding box regression and classification.
- **Data Augmentation**: Implements random scaling, shifting, hue adjustment, and saturation changes.
- **Custom Loss Function**: Implements the YOLOv1 loss function with adjustments for better performance.
- **Visualization Tools**: Includes functions to plot bounding boxes on images and visualize loss curves.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- [Anaconda](https://www.anaconda.com/) or `venv` for virtual environment management

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/deepdik/yolov1.git
   cd yolov1
   ```

2. **Create a Virtual Environment**

   Using Anaconda:

   ```bash
   conda create -n yolov1-env python=3.8
   conda activate yolov1-env
   ```

   Or using `venv`:

   ```bash
   python -m venv yolov1-env
   source yolov1-env/bin/activate  # On Windows use yolov1-env\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If a `requirements.txt` file is not provided, install the necessary packages manually:*

   ```bash
   pip install torch torchvision tqdm matplotlib pillow numpy
   ```

## Dataset Preparation

The script automatically downloads and prepares the Pascal VOC 2007 dataset. Ensure that you have enough disk space and a stable internet connection.

- The dataset will be downloaded to the `data` directory.

## Project Structure

```

├── models/                 # Directory to save trained models
├── results/                # Directory to save output images with bounding boxes
├── yolo_v1.py                 # Main script containing all the code
├── README.md               # This README file
└── requirements.txt        # Python package dependencies
```

## Usage

All functionalities are encapsulated within the `yolo_v1.py` script. You can control which part of the code to execute by uncommenting the corresponding function call in the `if __name__ == '__main__':` block.

### Training the Model

1. **Uncomment the Training Function**

   Open `yolo_v1.py` and ensure the `train_and_save_model()` function call is uncommented:

   ```python
   if __name__ == '__main__':
       train_and_save_model()    # Train the model
       # plot_test_images()        # Plot predictions on test images
       # plot_loss_graphs()        # Plot loss curves
   ```

2. **Run the Script**

   ```bash
   python yolo_v1.py
   ```

3. **Monitor Training**

   - The model will be trained for 135 epochs.
   - Training and testing losses will be saved in the `models/yolo_v1/<timestamp>/` directory.
   - You can use TensorBoard to monitor the training process:

     ```bash
     tensorboard --logdir runs
     ```

### Plotting Test Images

1. **Uncomment the Plotting Function**

   ```python
   if __name__ == '__main__':
       # train_and_save_model()    # Train the model
       plot_test_images()        # Plot predictions on test images
       # plot_loss_graphs()        # Plot loss curves
   ```

2. **Ensure Model Weights are Available**

   - Make sure you have trained the model or have the trained weights saved in `models/yolo_v1/final`.

3. **Run the Script**

   ```bash
   python yolo_v1.py
   ```

4. **View Results**

   - The output images with predicted bounding boxes will be saved in the `results` directory.

### Plotting Loss Curves

1. **Uncomment the Loss Plotting Function**

   ```python
   if __name__ == '__main__':
       # train_and_save_model()    # Train the model
       # plot_test_images()        # Plot predictions on test images
       plot_loss_graphs()        # Plot loss curves
   ```

2. **Ensure Loss Data is Available**

   - Make sure you have the `train_losses.npy` and `test_losses.npy` files saved in the project directory.

3. **Run the Script**

   ```bash
   python yolo_v1.py
   ```

4. **View the Plot**

   - A window will open displaying the training and testing loss curves over epochs.

## Code Explanation

### Model Architecture

- **Backbone Network**: ResNet-50 pre-trained on ImageNet.
- **Detection Layers**: Added convolutional and fully connected layers to predict bounding boxes and class probabilities.

```python
class YOLOv1ResNet(nn.Module):
    # Model definition
```

### Custom Loss Function

- **SumSquaredErrorLoss**: Implements the YOLOv1 loss function with separate components for localization, confidence, and classification losses.

```python
class SumSquaredErrorLoss(nn.Module):
    # Loss function implementation
```

### Dataset Class

- **YoloPascalVocDataset**: Custom dataset class that loads images and annotations, applies transformations, and constructs the ground truth tensor.

```python
class YoloPascalVocDataset(Dataset):
    # Dataset class implementation
```

## Results

After training, you should observe:

- **Loss Curves**: Training and testing losses decreasing over epochs.
- **Predicted Bounding Boxes**: Images in the `results` directory with bounding boxes and class labels drawn around detected objects.

## Supplemental Material

Below are additional resources related to the project:

- **Final Trained Model After 135 Epoch:** [Download here](https://drive.google.com/file/d/1deHiqN0u3FdV6eujv96NHbtbQFegOh0Y/view?usp=sharing)
- **Final Output Images (2.5K+):** [Download here](https://drive.google.com/file/d/1tXYrmGvKuPqDd9OR2utW8fFtHqlEJ_8n/view?usp=sharing)
- **Folder Architecture After Successful Run:** [Download here](https://drive.google.com/file/d/1uP9JZt2hgABN5w8xXbGRvk9WHfXFaGO7/view?usp=sharing)
- **Successful Run 135 Epoch:** [Download here](https://drive.google.com/file/d/1he7GoZU2--rVkgNxAYoJnBPuriJzidy9/view?usp=sharing)


## Troubleshooting

- **Out of Memory Errors**: Reduce the `BATCH_SIZE` if you encounter CUDA out-of-memory errors.
- **Dataset Download Issues**: Ensure you have a stable internet connection for downloading the Pascal VOC dataset.
- **Incorrect Class Predictions**: Make sure the class mapping (`VOC_CLASSES`) is correctly defined and matches the dataset labels.

## References

- **YOLOv1 Paper**: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- **PyTorch Documentation**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **Pascal VOC Dataset**: [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)
- **ResNet Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

