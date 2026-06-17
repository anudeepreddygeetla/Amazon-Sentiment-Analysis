import streamlit as st
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_resources():
    model = load_model('best_amazon_sentiment_model.keras')
    with open('amezon_sentiment_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

MAX_LEN = 100

def Clean_Text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

def predict_statement(text):
    clean = Clean_Text(text)
    sequence = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    prob = model.predict(padded)[0][0]
    sentiment = 'Positive' if prob > 0.5 else 'Negative'
    return sentiment, float(prob)

# UI
st.title("Amazon Review Sentiment Analyzer")
st.write("Type any product review — the model will predict Positive or Negative.")

user_input = st.text_area("Enter your review here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        sentiment, prob = predict_statement(user_input)
        if sentiment == "Positive":
            st.success(f"Sentiment: ✅ {sentiment}")
        else:
            st.error(f"Sentiment: ❌ {sentiment}")
        confidence = prob if sentiment == "Positive" else 1 - prob
        st.write(f"Confidence: {confidence:.2f}")