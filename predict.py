import joblib
import pandas as pd
from preprocessing import preprocess_text, read_stopwords, ukrainian_stopwords_file_path
from nltk.stem import SnowballStemmer

# Load the saved model
model = joblib.load('model.pkl')

# Initialize the SnowballStemmer
stemmer = SnowballStemmer("russian")

# Load Ukrainian stopwords
ukrainian_stopwords = read_stopwords(ukrainian_stopwords_file_path)


def predict_bias(sentence):
    # Preprocess the input sentence
    preprocessed_sentence = preprocess_text(
        sentence, stemmer, ukrainian_stopwords)

    # Transform the preprocessed sentence into a DataFrame
    input_data = pd.DataFrame(
        [preprocessed_sentence], columns=['preprocessed_text'])

    # Predict the bias category and its probability
    predicted_proba = model.predict_proba(input_data['preprocessed_text'])[0]
    predicted_label = model.predict(input_data['preprocessed_text'])[0]

    # Get the predicted bias category and its probability
    max_proba_index = predicted_proba.argmax()
    bias_category = model.named_steps['classifier'].classes_[max_proba_index]
    bias_probability = predicted_proba[max_proba_index]

    return bias_probability, bias_category


# Example usage:
sentence = "Your input sentence here"
probability, category = predict_bias(sentence)
print(
    f"The input sentence has a {probability * 100:.2f}% probability of being in the '{category}' category.")


"""
import sys
import joblib
import pandas as pd
from preprocessing import preprocess_text, read_stopwords, ukrainian_stopwords_file_path
from nltk.stem import SnowballStemmer

# Load the saved model
model = joblib.load(
    '/Users/korben/Documents/nulp/eighth_semester/nlp/bias_classifier/model.pkl')

# Initialize the SnowballStemmer
stemmer = SnowballStemmer("russian")

# Load Ukrainian stopwords
ukrainian_stopwords = read_stopwords(ukrainian_stopwords_file_path)


while True:

    def predict_bias(sentence):
        # Preprocess the input sentence
        preprocessed_sentence = preprocess_text(
            sentence, stemmer, ukrainian_stopwords)

        # Transform the preprocessed sentence into a DataFrame
        input_data = pd.DataFrame(
            [preprocessed_sentence], columns=['preprocessed_text'])

        # Predict the bias category and its probability
        predicted_proba = model.predict_proba(
            input_data['preprocessed_text'])[0]
        predicted_label = model.predict(input_data['preprocessed_text'])[0]

        # Get the predicted bias category and its probability
        max_proba_index = predicted_proba.argmax()
        bias_category = model.named_steps['classifier'].classes_[
            max_proba_index]
        bias_probability = predicted_proba[max_proba_index]

        return bias_probability, bias_category

    def safe_input(prompt):
        try:
            return input(prompt)
        except UnicodeDecodeError:
            print("Invalid characters detected. Please try again.\n")
            return safe_input(prompt)

    sentence = safe_input("The input: ")

    probability, category = predict_bias(sentence)
    print(
        f"Sentence has a {probability * 100:.2f}% probability of being in the '{category}' category.\n")
"""
