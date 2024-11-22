import streamlit as st
import joblib
import PyPDF2
import docx
import re

# Load the sentiment model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and remove whitespace
    return text

# Function to map the sentiment prediction to a label with neutral
def get_sentiment_label(probabilities, threshold=0.6):
    pos_prob = probabilities[1]
    if pos_prob > threshold:
        return "Positive"
    elif pos_prob < (1 - threshold):
        return "Negative"
    else:
        return "Neutral"

# Function to predict sentiment
def predict_sentiment(tweets, threshold=0.6):
    results = []
    for tweet in tweets:
        cleaned_tweet = clean_text(tweet)
        vectorized_tweet = vectorizer.transform([cleaned_tweet])
        probabilities = model.predict_proba(vectorized_tweet)[0]
        sentiment_label = get_sentiment_label(probabilities, threshold)
        results.append((cleaned_tweet, sentiment_label))
    return results

# Streamlit UI
st.title("Sentiment Analysis App")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])

# Text area for manual input
text_input = st.text_area("Or manually enter tweets here (one tweet per line)")

# Analyze button
if st.button("Analyze"):
    tweets = []
    
    # Check if a file is uploaded
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                tweets.append(page.extract_text())
            tweets = "\n".join(tweets).split('\n')  # Combine pages and split into lines
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Extract text from Word
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                tweets.append(para.text)
            tweets = [tweet for tweet in tweets if tweet]  # Filter out empty lines

    # Check if manual input is provided
    if text_input:
        tweets.extend(text_input.split('\n'))

    if tweets:
        # Analyze sentiment
        results = predict_sentiment(tweets)
        
        # Display results
        st.subheader("Results")
        for cleaned_tweet, sentiment in results:
            st.write(f"**Tweet**: {cleaned_tweet}  \n**Sentiment**: {sentiment}")
    else:
        st.warning("No tweets found for analysis.")
