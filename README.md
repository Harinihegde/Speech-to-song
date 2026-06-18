

```md
## Dataset

A custom dataset was created containing:

- Audio clips
- Corresponding labels

Each audio file is mapped to a class:

```

filename | label

```

Example:

```

audio1.wav | transforming
audio2.wav | non-transforming

```

---

## Project Structure

```

speech-to-song/

├── data/
│   ├── audio_dataset/
│   ├── labels.csv
│   └── features.csv
│
├── feature_extraction.py
│
└── README.md

````

---

## Installation

Clone the repository:

```bash
git clone <repository-link>
cd speech-to-song
````

Install dependencies:

```bash
pip install librosa numpy pandas scikit-learn
```

---

## Usage

Run feature extraction:

```bash
python feature_extraction.py
```

The extracted features will be stored in:

```
data/features.csv
```

This file can be used for training machine learning models.

---

## Learning Outcomes

Through this project, the following concepts were explored:

* Audio signal processing
* Feature engineering
* Machine learning data preparation
* Working with audio datasets
* Using Librosa for audio analysis

---

## Future Improvements

* Train and compare multiple ML models
* Increase dataset size
* Add real-time audio classification
* Experiment with deep learning approaches

---

## Author

Harini Hegde

```
```
