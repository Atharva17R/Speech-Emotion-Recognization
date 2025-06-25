# Speech-Emotion-Recognization

## Project Description

This project implements a deep learning-based Speech Emotion Recognition (SER) system that classifies emotions from short audio clips of speech. The model is trained to identify multiple emotions (such as neutral, calm, happy, sad, angry, fearful, disgust, and surprised) using advanced feature extraction and a convolutional neural network (CNN) and (LSTM) architecture. The system is designed for robust performance and can be used via a web interface or command-line scripts.

## Pre-processing Methodology

- **Defining feature functions** -defining and labelling emotionsinto integers and defining Function to extract emotion label from filename

  
- **Audio Standardization:**  All audio files are resampled to 16kHz and trimmed or padded to exactly 3 seconds to ensure consistency.

 
- **Visualize Waveform, Spectrogram, MFCCs**- visulaization of dataset to plot Spectrogram, MFCCs and waveform

  
- **preprocessing dataset for training** - further processing and splitting dataset for train and testing dataset

  
 - **Feature Extraction:**  
  For each audio file, two channels of features are extracted:
   **Channel 1:** Mel-spectrogram (32 Mel bands, dB scale)
   **Channel 2:** Stacked features including MFCCs (13 coefficients), chroma, and spectral contrast  
  Each channel is normalized (zero mean, unit variance) and shaped to (32, 94, 2).


-**Model training- CNN - LSTM** : We use a hybrid CNN-LSTM model to capture both spatial patterns in spectrograms and temporal patterns across time.

  **CNN layers** learn spatial (spectral) features
  **LSTM layers** learn sequential dependencies


- **Evaluation of model** - accuracy and loss graphs, confusion matrix and overall accuracy 

## Accuracy Metrics

| Metric                   | Value      | Status |
|--------------------------|------------|--------|
| **Validation Accuracy**  | 81.00%     | PASS   |
| **Weighted F1-score**    | 0.81       | PASS   |
| **All Class Accuracy**   | >75%       | PASS   |

- Detailed classification report and confusion matrix are generated after training in report.

### 📊 Classification Report

| Label      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Angry      | 0.90      | 0.85   | 0.88     | 75      |
| Calm       | 0.89      | 0.83   | 0.86     | 75      |
| Disgust    | 0.79      | 0.79   | 0.79     | 39      |
| Fearful    | 0.83      | 0.73   | 0.78     | 75      |
| Happy      | 0.97      | 0.77   | 0.86     | 75      |
| Neutral    | 0.68      | 0.95   | 0.79     | 38      |
| Sad        | 0.70      | 0.75   | 0.72     | 75      |
| Surprised  | 0.69      | 0.92   | 0.79     | 39      |
| **Accuracy**   |           |        | **0.81**  | **491** |
| **Macro Avg** | 0.81      | 0.82   | 0.81     | 491     |
| **Weighted Avg** | 0.83  | 0.81   | 0.81     | 491     |


![Confusion Matrix](https://github.com/Atharva17R/Speech-Emotion-Recognization/blob/main/download%20(1).png)
 
