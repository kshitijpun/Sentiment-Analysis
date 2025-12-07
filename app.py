import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords


# Streamlit Page Config

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")


# Load Stopwords Safely

@st.cache_resource
def load_stopwords():
    try:
        return stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        return stopwords.words("english")


# Load Model & Vectorizer

@st.cache_resource
def load_model_and_vectorizer():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# Text Preprocessing + Prediction

def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = " ".join(text)

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    return "Positive" if prediction == 1 else "Negative"


# Card UI

def create_card(tweet_text, sentiment):
    color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
    return f"""
    <div style="
        background-color:{color};
        padding:15px;
        border-radius:10px;
        margin-bottom:10px;
        color:white;
    ">
        <h4>{sentiment} Sentiment</h4>
        <p>{tweet_text}</p>
    </div>
    """


# Main App

def main():
    st.title("🐦 Twitter Sentiment Analysis ")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    option = st.selectbox(
        "Choose Input Type",
        ["Manual Text Input", "Demo Tweets"]
    )

  
    # Manual Text Analysis
    
    if option == "Manual Text Input":
        text_input = st.text_area("Enter text to analyze sentiment")

        if st.button("Analyze Sentiment"):
            if text_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.success(f"Sentiment: **{sentiment}**")

   
    # Demo Tweets Analysis
 
    else:
        st.info("Live Twitter scraping is disabled. Showing demo tweets.")
        demo_tweets = [
            "I love working on NLP projects!",
            "Streamlit makes ML apps so easy.",
            "Sometimes bugs are frustrating...",
            "Python is an amazing language.",
            "Learning machine learning is fun!"
        ]

        for tweet_text in demo_tweets:
            sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
            st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)


# Run App

if __name__ == "__main__":
    main()
