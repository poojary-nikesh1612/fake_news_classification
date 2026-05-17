# Fake News Detection using spaCy Word Embeddings

This project performs fake news classification using pretrained spaCy word embeddings and machine learning algorithms.

The text data is converted into dense vector representations using the `en_core_web_lg` spaCy model, and classification is performed using:

- Multinomial Naive Bayes
- K-Nearest Neighbors (KNN)

---

## Features

- NLP preprocessing using spaCy
- Pretrained word embeddings (`en_core_web_lg`)
- Feature scaling using MinMaxScaler
- Classification using:
  - Multinomial Naive Bayes
  - KNN Classifier
- Model evaluation using classification report

---

## Technologies Used

- Python
- pandas
- NumPy
- spaCy
- scikit-learn

---

## Dataset

Dataset used:

`Fake_Real_Data.csv`

The dataset contains:
- News text
- Labels:
  - Fake
  - Real

---

## Installation

Install required libraries:

```bash
pip install pandas numpy spacy scikit-learn
```

Download spaCy large English model:

```bash
python -m spacy download en_core_web_lg
```

---

## Project Workflow

1. Load dataset
2. Convert labels into numerical values
3. Generate word embeddings using spaCy
4. Convert embeddings into feature matrix
5. Scale features using MinMaxScaler
6. Split data into train and test sets
7. Train:
   - Multinomial Naive Bayes
   - KNN
8. Evaluate models
---

## Evaluation Metric

Classification performance is evaluated using:

- Precision
- Recall
- F1-score
- Accuracy
