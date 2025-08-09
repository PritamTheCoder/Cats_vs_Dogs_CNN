# Cats vs Dogs CNN Classifier

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs. It includes data loading, model training, evaluation, and a simple Flask web application for inference.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset Structure](#dataset-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Training the Model](#training-the-model)
* [Testing on Single Images](#testing-on-single-images)
* [Running the Web Application](#running-the-web-application)
* [Project Structure](#project-structure)
* [Acknowledgements](#acknowledgements)

---

## Project Overview

This project builds a CNN to distinguish between cat and dog images. It uses PyTorch for deep learning and torchvision for dataset management. The model is trained on images organized in folders, evaluated on a test split, and saved for inference.

A Flask-based web app allows users to upload images and get real-time predictions.

---

## Dataset Structure

Your image dataset should be organized as follows:

```
data/
└── train/
    ├── cats/
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    │   └── ...
    └── dogs/
        ├── dog1.jpg
        ├── dog2.jpg
        └── ...
```

* Place all training images in `data/train` under folders `cats` and `dogs`.
* Images should be RGB format and of reasonable size.

---

## Requirements

* Python 3.7+
* PyTorch
* torchvision
* scikit-learn
* Flask (for web app)
* Pillow
* Other common Python libraries

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/PritamTheCoder/Cats_vs_Dogs_CNN.git
   cd Cats_vs_Dogs_CNN
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install torch torchvision scikit-learn pillow flask
   ```

---

## Training the Model

Run the training script:

```bash
python train.py
```

This will:

* Load and preprocess the dataset from `data/train`.
* Train the CNN model for 5 epochs (you can modify this in the script).
* Evaluate accuracy on a held-out test set.
* Save the trained model as `cat_v_dog_cnn.pth`.

**Tips:**

* Make sure your dataset directory exists and is correctly structured.
* Use GPU if available for faster training.

---

## Testing on Single Images

To test a single image with the trained model:

```bash
python test_img.py path/to/your/image.jpg
```

This script will load the saved model, preprocess the input image, run inference, and print the prediction with confidence.

---

## Running the Web Application

The web app allows you to upload an image and get a prediction through a user-friendly interface.

1. Run the Flask app:

```bash
python app.py
```

2. Open your browser and navigate to `http://127.0.0.1:5000`

3. Upload an image of a cat or dog, and see the prediction displayed.

---

## Project Structure

```
.
├── data/
│   └── train/
│       ├── cats/
│       └── dogs/
├── app.py                 # Flask web app
├── train.py               # Training script
├── test_img.py            # Single image inference script
├── network.py             # Model architecture
├── ImageLoader.py         # (Optional) custom dataset loader (not required with current train.py)
├── cat_v_dog_cnn.pth      # Saved model weights (after training)
└── README.md              # This file
```

---

## Acknowledgements

* Uses PyTorch and torchvision libraries.
* Inspired by common Cats vs Dogs classification tutorials.
* Thanks to the [Dogs vs Cats dataset on Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).

