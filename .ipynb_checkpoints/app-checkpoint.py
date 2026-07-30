import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('model.h5')
tokenizer = joblib.load('tokenizer.pkl')

def predict_sentiment(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive 😀" if prediction > 0.5 else "Negative 😞"
    return sentiment, prediction

st.title("🎬  Review Sentiment Analysis")
review = st.text_area("Enter a movie review:")
if st.button("Analyze"):
    if review.strip():
        sentiment, score = predict_sentiment(review)
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Confidence score: {score:.3f}")
    else:
        st.warning("Please enter a review first.")