import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


nltk.download('stopwords')
file_path = '/Users/korben/Documents/nulp/eighth_semester/nlp/bias_classifier/preprocessed_dataset.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')


def preprocess_text(text, stemmer, stopwords):
    if not isinstance(text, str):
        return None

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^а-яїієґ\s]', '', text)

    # Tokenize the text
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stopwords]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    # Reconstruct the text
    preprocessed_text = ' '.join(words)

    return preprocessed_text


def read_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    return set(stopwords)


ukrainian_stopwords_file_path = '/Users/korben/Documents/nulp/eighth_semester/nlp/bias_classifier/stopwords_ua.txt'
ukrainian_stopwords = read_stopwords(ukrainian_stopwords_file_path)

# Initialize the SnowballStemmer
stemmer = SnowballStemmer("russian")

# Apply the preprocessing function to the text column using the Ukrainian stopwords
data['preprocessed_text'] = data['text'].apply(
    lambda x: preprocess_text(x, stemmer, ukrainian_stopwords))

# Save the preprocessed dataset to a file
output_file_path = '/Users/korben/Documents/nulp/eighth_semester/nlp/bias_classifier/preprocessed_dataset.xlsx'
data.to_excel(output_file_path, engine='openpyxl', index=False)
