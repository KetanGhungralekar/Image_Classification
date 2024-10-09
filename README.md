# Handwritten Digit Classification using Deep Nueral Networks on GPU

This repository contains a nueral network model implemented in Python for classifying handwritten digits from the MNIST dataset. The model achieves an accuracy of **98.11%** and aims to provide insights into basic machine learning concepts using Deep Nueral Network.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Description](#model-description)
- [Getting Started](#getting-started)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Introduction

The MNIST dataset is a classic dataset in the field of machine learning and image processing, containing 60,000 training images and 10,000 testing images of handwritten digits. This project demonstrates how to implement a deep nueral network model to classify these digits effectively.

## Requirements

To run this project, ensure you have the following packages installed:

- Python 3.x
- NumPy
- Pandas
- PyTorch
- torchvision

You can install the required packages using pip:

```bash
pip install numpy pandas torch torchvision
```

## Dataset

The dataset used for this project is the **MNIST** dataset, which can be directly accessed through the `torchvision` library. It consists of 28x28 pixel images of handwritten digits (0-9).

## Model Description

This project employs a nueral network model, which is a linear model used for binary classification. In this case, we extend it for multi-class classification by applying the softmax function to output probabilities for each digit class.

The model architecture includes:

- Input Layer: Takes in the 784 pixel values (28x28 images flattened)
- Output Layer: Produces probability scores for each of the 10 digit classes (0-9)

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd Image_Classification
    ```

2. Open the Jupyter Notebook:

    ```bash
    jupyter notebook main.ipynb
    ```

3. Run the notebook cells to train the model and evaluate its performance on the test set.

## Results

The model achieved an accuracy of **98.11%** on the test dataset, demonstrating its effectiveness in classifying handwritten digits.

## Future Work

To improve the model's performance further, the following enhancements are planned:

- Increasing the number of hidden layers in the model.
- Adjusting the layer sizes for better feature extraction.
- Exploring other algorithms for comparison.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
