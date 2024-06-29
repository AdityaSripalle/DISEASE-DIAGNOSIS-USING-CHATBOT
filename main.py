import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Step 1: Load and Preprocess Data
data = {
    'symptoms': [
        'fever headache nausea',
        'cough sore_throat fever',
        'headache fatigue muscle_pain',
        'fever rash joint_pain',
        'cough fever difficulty_breathing',
        'sore_throat headache runny_nose'
    ],
    'disease': ['flu', 'cold', 'flu', 'chickenpox', 'covid', 'cold']
}

df = pd.DataFrame(data)

# Basic text preprocessing
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

df['symptoms'] = df['symptoms'].apply(preprocess_text)

# Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['symptoms']).toarray()
y = df['disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train a Predictive Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 3: Build the Chatbot
def chatbot_response(user_input):
    # Preprocess the input
    user_input = preprocess_text(user_input)
    user_input_vec = vectorizer.transform([user_input]).toarray()
    
    # Make a prediction
    prediction = model.predict(user_input_vec)
    return prediction[0]

# Example interaction
print("Welcome to the disease diagnosis chatbot!")
while True:
    user_input = input("Please describe your symptoms: ")
    if user_input.lower() == 'exit':
        break
    diagnosis = chatbot_response(user_input)
    print(f"Our preliminary diagnosis is: {diagnosis}")
