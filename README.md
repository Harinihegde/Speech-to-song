# 🎵 Speech-to-Song Classification | Machine Learning

## Overview

This project focuses on classifying audio clips into two categories:

- **Transforming**
- **Non-Transforming**

The project uses audio signal processing techniques to extract meaningful features from speech clips and applies machine learning methods for classification.

Since a suitable public dataset was unavailable, a **custom dataset** of audio clips was created for this project.

---

## Objective

The goal of this project is to analyze audio characteristics and identify patterns that distinguish transforming and non-transforming speech clips.

The workflow includes:

1. Collecting and labeling audio samples
2. Extracting audio features
3. Creating a structured feature dataset
4. Training a machine learning classifier

---

## Technologies Used

- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn

---

## Feature Extraction

Audio features were extracted using the **Librosa** library. A total of **22 audio features** are generated from each audio clip.

### 1. MFCC Features

Mel Frequency Cepstral Coefficients (MFCCs) capture the characteristics of the audio signal.

- 13 MFCC coefficients

### 2. Pitch Features

Pitch-related features extracted using frequency analysis.

- Mean pitch
- Pitch variation
- Pitch range

### 3. Rhythm Features

Features related to timing and beats.

- Tempo
- Onset count

### 4. Spectral Features

Spectral properties of the audio signal.

- Spectral centroid
- Spectral rolloff
- Zero crossing rate
- RMS energy

---

## Dataset

A custom dataset was created containing audio clips and corresponding labels. Each audio file is mapped to a class:

```
filename         | label
audio1.wav       | transforming
audio2.wav       | non-transforming
```

---

## Project Structure

```
speech-to-song/
│
├── data/
│   ├── audio_dataset/
│   ├── labels.csv
│   └── features.csv
│
├── feature_extraction.py
│
└── README.md
```

---

## Installation

Clone the repository:

```
git clone <repository-link>
cd speech-to-song
```

Install dependencies:

```
pip install librosa numpy pandas scikit-learn
```

---

## Usage

Run feature extraction:

```
python feature_extraction.py
```

The extracted features will be stored in `data/features.csv`, which can be used for training machine learning models.

---

## Learning Outcomes

Through this project, the following concepts were explored:

- Audio signal processing
- Feature engineering
- Machine learning data preparation
- Working with audio datasets
- Using Librosa for audio analysis

---

## Future Improvements

- Train and compare multiple ML models
- Increase dataset size
- Add real-time audio classification
- Experiment with deep learning approaches

---

## Author

**Harini Hegde**
