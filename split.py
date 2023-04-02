import pandas as pd
from sklearn.model_selection import train_test_split

# Load the preprocessed dataset
file_path = '/Users/korben/Documents/nulp/eighth_semester/nlp/bias_classifier/preprocessed_dataset.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Define the features (preprocessed text) and the target (labels)
X = data['preprocessed_text']
# Replace 'label' with the actual column name containing the bias categories in your dataset
y = data['label']

# Split the dataset (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
