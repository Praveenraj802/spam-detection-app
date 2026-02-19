import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set NLTK data path for Render
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

app = Flask(__name__)

# Load and train model on startup
print("Loading dataset and training model...")
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data['clean_message'] = data['message'].apply(clean_text)

# Train model
X = data['clean_message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("Model trained successfully!")
print(f"Training accuracy: {model.score(X_train_vec, y_train):.2%}")
print(f"Test accuracy: {model.score(X_test_vec, y_test):.2%}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message.strip():
            return jsonify({'error': 'Please enter a message'}), 400
        
        # Clean and predict
        cleaned_message = clean_text(message)
        message_vec = vectorizer.transform([cleaned_message])
        prediction = model.predict(message_vec)[0]
        probability = model.predict_proba(message_vec)[0]
        
        result = {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': float(probability[prediction]) * 100,
            'spam_probability': float(probability[1]) * 100,
            'ham_probability': float(probability[0]) * 100
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n[INFO] Flask Spam Detection App Starting...")
    print("[LINK] Open your browser and go to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
