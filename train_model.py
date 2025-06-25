import pandas as pd
import re
import joblib
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_FILE = os.path.join('data', 'crypto_currency_sentiment_dataset.csv')
MODEL_DIR = 'saved_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_classifier.joblib')

def preprocess_text(text):
    """
    Cleans and preprocesses a single piece of text.
    - Converts to lowercase
    - Removes URLs, numbers, and punctuation
    - Removes stopwords
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphabetic characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def train_and_evaluate():
    """
    Loads data, preprocesses it, builds a TF-IDF + Logistic Regression pipeline,
    evaluates it using cross-validation, and saves the final trained model.
    """
    # --- Load and Prepare Data ---
    try:
        df = pd.read_csv(DATA_FILE)
        df.dropna(subset=['Comment', 'Sentiment'], inplace=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE}'. Please run check the path.")
        return

    print("Preprocessing text data...")
    df['cleaned_comment'] = df['Comment'].apply(preprocess_text)

    X = df['cleaned_comment']
    y = df['Sentiment']

    # --- Build Model Pipeline ---
    # This pipeline combines feature extraction (TF-IDF) and the classifier.
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, C=1.0, solver='liblinear'))
    ])

    # --- Evaluate Model with Cross-Validation ---
    print("\nEvaluating model with 5-fold stratified cross-validation...")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    cv_results = cross_validate(pipeline, X, y, cv=cv_strategy, scoring=scoring_metrics)

    print("\n--- Cross-Validation Results ---")
    print(f"Average Accuracy:  {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
    print(f"Average Precision: {cv_results['test_precision_macro'].mean():.4f}")
    print(f"Average Recall:    {cv_results['test_recall_macro'].mean():.4f}")
    print(f"Average F1-Score:  {cv_results['test_f1_macro'].mean():.4f}")
    print("---------------------------------")


    # --- Train Final Model and Save ---
    print("\nTraining final model on the entire dataset...")
    pipeline.fit(X, y)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model successfully trained and saved to '{MODEL_PATH}'")

if __name__ == '__main__':
    # Ensure you have the stopwords package
    try:
        stopwords.words('english')
    except LookupError:
        import nltk
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    train_and_evaluate()
