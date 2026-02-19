import streamlit as st
import pandas as pd
import nltk
import joblib
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords only if not already present
import os
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

data['message'] = data['message'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# -------- STREAMLIT UI --------
st.title("📧 Spam Email Detection System")
st.write("Enter an email message to check whether it is **Spam or Not Spam**")

user_input = st.text_area("✉️ Email Content")

if st.button("Check Spam"):
    if user_input.strip() == "":
        st.warning("Please enter an email message!")
    else:
        cleaned_input = clean_text(user_input)
        vector_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vector_input)

        if prediction[0] == 1:
            st.error("❌ Spam Email Detected")
        else:
            st.success("✅ This is a Safe Email")
