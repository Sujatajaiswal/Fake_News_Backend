from flask import Flask, request, jsonify  # type: ignore
from flask_cors import CORS  # type: ignore
import pandas as pd  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.naive_bayes import MultinomialNB  # type: ignore
import joblib  # type: ignore
import os

app = Flask(__name__)
CORS(app)

# Load your pre-trained model and vectorizer
model = joblib.load('model.pkl')  # Make sure to create and save your model
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return "Fake News Detector Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)

    prediction_result = int(prediction[0])

    return jsonify({'prediction': prediction_result})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
