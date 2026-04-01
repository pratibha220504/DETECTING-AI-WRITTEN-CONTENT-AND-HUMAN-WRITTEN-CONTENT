import streamlit as st
import joblib
import re


# Load Model and Vectorizer
model = joblib.load("logistic_model.sav")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Text Cleaning Function
def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Prediction Function
def predict_text(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector).max()

    return prediction, probability


# Streamlit UI
st.title("AI vs Human Text Classification")

user_input = st.text_area("Enter Text")


if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter text")

    else:

        prediction, probability = predict_text(user_input)

        st.write("Prediction:", prediction)

        st.write("Confidence:", round(probability * 100, 2), "%")