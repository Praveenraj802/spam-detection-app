import pandas as pd
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1','v2']]
data.columns = ['label','message']
data['label'] = data['label'].map({'ham':0, 'spam':1})

print("Dataset Loaded Successfully!")
print(data.head())

# NLP Cleaning
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data['clean_message'] = data['message'].apply(clean_text)

# Split
X = data['clean_message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test
y_pred = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# User input
while True:
    msg = input("\nEnter email message (type exit to stop): ")

    if msg.lower() == "exit":
        break

    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    result = model.predict(msg_vec)

    if result[0] == 1:
        print("[SPAM] This is a spam message!")
    else:
        print("[HAM] This is a normal message")
