import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load the preprocessed dataset
file_path = '/Users/korben/Documents/nulp/eighth_semester/nlp/bias_classifier/preprocessed_dataset.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Drop rows with NaN values
data.dropna(subset=['text'], inplace=True)

# Or alternatively, fill NaN values with an empty string
data['text'].fillna('', inplace=True)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    data['preprocessed_text'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the model using the training set
pipeline.fit(X_train, y_train)

# Evaluate the model's performance on the validation set
y_pred = pipeline.predict(X_val)
print(classification_report(y_val, y_pred))

# Save the trained model for future use
joblib.dump(
    pipeline, '/Users/korben/Documents/nulp/eighth_semester/nlp/bias_classifier/model.pkl')
