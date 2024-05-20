# EEG Truth-Lie Classification

This repository contains code for classifying EEG data to determine whether a subject is telling the truth or lying. The code uses a Convolutional Neural Network (CNN) model to perform the classification.

## Table of Contents
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Running on Kaggle](#running-on-kaggle)
- [Requirements](#requirements)

## Dataset

The dataset consists of EEG recordings from 27 subjects, each with two sessions:
1. One session where the subject is telling the truth.
2. One session where the subject is lying.

The data is organized into two folders within the `truth-lie` directory:
- `truth-lie/truth`: Contains CSV files for the truth sessions.
- `truth-lie/lie`: Contains CSV files for the lie sessions.

Each CSV file contains 5 columns representing EEG channels (`EEG.AF3`, `EEG.T7`, `EEG.Pz`, `EEG.T8`, `EEG.AF4`) and 9600 rows, sampled at 128 Hz.

## Preprocessing

1. **Loading Data**: All CSV files from the truth and lie folders are loaded and concatenated into separate DataFrames.
2. **Normalization**: The data is normalized using `StandardScaler`.
3. **Segmentation**: The continuous time series data is segmented into smaller windows of 128 samples with a 50% overlap.

## Feature Extraction

**Discrete Wavelet Transform (DWT)**: Features are extracted using the DWT method with the 'db4' wavelet at level 4.

## Model

A CNN model is defined with the following architecture:
- Three stages of Convolutional, Batch Normalization, MaxPooling, and Dropout layers.
- Flatten layer.
- Three Fully Connected (Dense) layers with Dropout.
- Output layer with Sigmoid activation for binary classification.

## Training

The model is trained using:
- `binary_crossentropy` loss.
- `adam` optimizer.
- Early stopping and model checkpointing to save the best model.

## Evaluation

The model is evaluated using accuracy, F1-score, and a classification report.

## Usage

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/eeg-truth-lie-classification.git
    cd eeg-truth-lie-classification
    ```

2. **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Code**:
    Ensure the dataset is placed in the `truth-lie` folder as described. Then run the main script:
    ```bash
    python main.py
    ```

## Running on Kaggle

1. **Upload the Dataset**:
    - Go to Kaggle and create a new notebook.
    - Upload the `truth-lie` folder containing your CSV files into the Kaggle notebook environment.

2. **Copy the Code**:
    - Copy the contents of `main.py` from this repository and paste it into a cell in the Kaggle notebook.

3. **Install Dependencies**:
    - Add a cell at the beginning of the notebook to install the required Python packages:
    ```python
    !pip install pandas numpy scikit-learn pywt tensorflow
    ```

4. **Run the Notebook**:
    - Run all the cells in the notebook to execute the code.

## Requirements

The following Python packages are required:
- pandas
- numpy
- scikit-learn
- pywt
- tensorflow

You can install them using:
```bash
pip install pandas numpy scikit-learn pywt tensorflow
