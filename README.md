# 🧠 Speech-Based Dementia Detection using Wav2Vec2.0

This project focuses on detecting various stages of dementia using a multi-label classification model trained on raw speech data. Leveraging Wav2Vec2.0 and the Hugging Face ecosystem, we fine-tune speech representations and apply advanced data preprocessing and augmentation techniques to improve performance on small medical datasets.

---

## 📝 Problem Statement

Dementia affects cognitive functions such as memory, language, and reasoning. Early detection through non-invasive means like speech signals can aid in clinical diagnosis and timely intervention. This project aims to build a deep learning pipeline to classify patients into dementia categories using only their spoken audio recordings.

---

## 🎯 Objectives

- Structure a multi-label dataset for speech-based dementia classification.
- Fine-tune Wav2Vec2.0 on dementia-specific audio using Hugging Face Transformers.
- Address class imbalance and generalization issues through SpecAugment and MixSpeech.
- Evaluate model performance using classification metrics like accuracy, precision, recall, and F1-score.

---

## 🧪 Dataset

The dataset is derived from the **DementiaNet** metadata file and includes speech samples of demented and non-demented individuals.

| Dataset                | Labels                                  | Format     |
|------------------------|-----------------------------------------|------------|
| `train_dm.csv`         | Dementia / No Dementia                  | Binary     |
| `valid_dm.csv`         | Dementia / No Dementia                  | Binary     |
| `train_dm_mul.csv`     | Five / Ten / Zero years to dementia, No Dementia | Multi-label |
| `valid_dm_mul.csv`     | Five / Ten / Zero years to dementia, No Dementia | Multi-label |

All audio was resampled to 16kHz and converted to mono format. Missing audio files were handled by injecting 1-second silent samples.

---

## 🧱 Project Structure

```bash
dementia-detection-wav2vec2/
├── data/                     # CSVs and metadata
├── notebooks/                # Jupyter notebooks (training + evaluation)
├── report/                   # Midterm/Final writeups
├── requirements.txt
├── README.md
└── .gitignore
