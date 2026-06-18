# Amazon Product Review — Sentiment Analysis
## 🚀 Live Demo
👉 [Try the app here](https://huggingface.co/spaces/GeetlaAnudeepReddy/Sentiment-Analysis)
### Comparing RNN, LSTM, GRU, BiLSTM, and BiGRU from Scratch

---

## What This Project Does

This project takes Amazon product reviews and predicts whether a review is **Positive** or **Negative** — using deep learning models built from scratch, without any pretrained models.

Given a sentence like:
> *"I Ate This chocolate which is just like a toothpaste taste"*

The model outputs: **Negative (confidence: 0.21)**

---

## Dataset

**Amazon Fine Food Reviews** — sourced from Kaggle
- 568,000+ real customer reviews
- Ratings from 1 to 5 stars
- Link: [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

---

## The Approach

### Data Decisions
- Removed 3-star reviews — they're genuinely ambiguous, neither clearly positive nor negative
- 1–2 stars → **Negative (0)**, 4–5 stars → **Positive (1)**
- Dataset was heavily imbalanced (positive reviews dominate), so downsampling was applied to balance classes
- Sampled 25,000 reviews for training — enough to learn patterns without crashing a local machine

### Text Preprocessing
- Lowercasing, removing HTML tags, special characters, and extra spaces
- Tokenization → Sequences → Padding (MAX_LEN = 100)
- Vocabulary size: 10,000 most frequent words

### Why Not Use BERT?
BERT and transformer-based models are the production standard — but they require significant GPU memory and compute. This project focuses on understanding sequential architectures from the ground up. BiLSTM/BiGRU are a natural stepping stone before moving to transformers.

---

## Models Compared

| Model | Val Accuracy | Notes |
|-------|-------------|-------|
| Simple RNN | ~87% | Unstable training, gradient issues |
| LSTM | ~86% | More stable than RNN |
| **GRU** | **~89%**  | Best performer — simpler, faster | [Beter Performance]
| Stacked BiLSTM | ~86% | Bidirectional but no gain here |
| Stacked BiGRU | ~86.4% | Deeper didn't help |

### Why GRU Won
Amazon reviews are mostly direct — *"great product"*, *"terrible quality"*. GRU's simpler gating mechanism was enough to capture these patterns. Bidirectional models need more data and longer sequences to show their advantage. With 25k samples and max length 100, simple GRU was the sweet spot.

### Where the Models Struggle
All sequential models struggle with contrast sentences like:
> *"The packaging was terrible but the product itself is amazing"*

The model sees "terrible" early and that signal dominates. This is a fundamental limitation of left-to-right sequential reading — BERT handles this better because it reads both directions simultaneously.

---

## Hyperparameter Tuning

Used **Random Search** across 10 trials to find the best config for the first notebook:

```
Search Space:
- Models: ['RNN', 'LSTM', 'GRU']
- Embedding dim: [32, 64, 128]
- Units: [64, 128]
- Dropout rate: [0.2, 0.3, 0.4, 0.5]
- Learning rate: [0.001, 0.1]
```

Best config found: `GRU | embedding_dim=64 | units=64 | dropout=0.4 | lr=0.001`

---

## Training Setup

```python
callbacks = [
    ModelCheckpoint('best_amazon_sentiment_model.keras', monitor='val_accuracy', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]
```

EarlyStopping was key — every model started overfitting after 4–5 epochs (train accuracy hitting 99% while val plateaued). EarlyStopping restored weights from the best epoch automatically.

---

## Real World Prediction

```python
def predict_statement(text):
    clean = Clean_Text(text)
    sequence = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    prob = model.predict(padded)[0][0]
    sentiment = 'Positive' if prob > 0.5 else 'Negative'
    return sentiment, prob

predict_statement("This product is absolutely amazing!")
# → ('Positive', 0.94)

predict_statement("Tastes like medicine, completely disappointed")  
# → ('Negative', 0.18)
```

---

## Project Structure

```
Amazon-Sentiment-Analysis/
├── Sentimental-Analysis.ipynb            # RNN, LSTM, GRU + Hyperparameter Tuning
├── Sentimental-Analysis-BiLSTM-BiGRU.ipynb  # BiLSTM and BiGRU comparison
├── best_amazon_sentiment_model.keras     # Saved GRU model (best performer)
├── amezon_sentiment_tokenizer.pkl        # Tokenizer (required for inference)
└── README.md
```

---

## How to Run Locally

```bash
git clone https://github.com/anudeepreddygeetla/Amazon-Sentiment-Analysis.git
cd Amazon-Sentiment-Analysis
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and update the CSV path in the notebook.

To use the saved model directly:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model('best_amazon_sentiment_model.keras')
with open('amezon_sentiment_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
```

---

## Tech Stack

`Python` `TensorFlow/Keras` `NumPy` `Pandas` `Matplotlib` `Seaborn` `Scikit-learn`

---

## Key Learnings

- Class imbalance in NLP datasets is real — downsampling changed results significantly
- More complex architecture ≠ better results. GRU outperformed stacked BiGRU
- EarlyStopping is not optional — every model here overfitted without it
- Vocabulary size matters. Low vocab (1000) maps most meaningful words to OOV tokens
- Sequential models have a fundamental weakness with contrast sentences — that's what transformers solve

---

*Built from scratch on a local machine. No pretrained models, no cloud GPU.*
