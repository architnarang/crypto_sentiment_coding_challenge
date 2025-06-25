import pandas as pd
import re
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_FILE = os.path.join('data', 'crypto_currency_sentiment_dataset.csv')
MODEL_DIR = 'saved_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_classifier.joblib')

# --- Initialize Lemmatizer ---
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Cleans, preprocesses, and lemmatizes a single piece of text.
    - Converts to lowercase
    - Removes URLs, numbers, and punctuation
    - Tokenizes text
    - Lemmatizes tokens
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
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize and remove stopwords
    stop_words = set(stopwords.words('english'))
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(lemmatized_tokens)

def train_and_evaluate():
    """
    Loads data, preprocesses it, and uses GridSearchCV to find the best
    hyperparameters for a TF-IDF + Logistic Regression pipeline. It then
    evaluates and saves the best model.
    """
    # --- Load and Prepare Data ---
    try:
        df = pd.read_csv(DATA_FILE)
        df.dropna(subset=['Comment', 'Sentiment'], inplace=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE}'. Please check the path.")
        return

    print("Preprocessing text data with lemmatization...")
    df['cleaned_comment'] = df['Comment'].apply(preprocess_text)

    X = df['cleaned_comment']
    y = df['Sentiment']

    # --- Build Model Pipeline ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000))
    ])

    # --- Define Hyperparameter Grid for GridSearchCV ---
    # We will test different values for TF-IDF and Logistic Regression parameters.
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)], # Test unigrams and bigrams
        'tfidf__max_df': [0.75, 1.0],           # Ignore words that are too frequent
        'tfidf__min_df': [1, 5],                # Ignore words that are too rare
        'classifier__C': [0.1, 1, 10]           # Test different regularization strengths
    }

    # --- Set up GridSearchCV ---
    # It will test all parameter combinations using 5-fold cross-validation.
    # n_jobs=-1 uses all available CPU cores to speed up the search.
    print("\nStarting GridSearchCV to find the best model parameters...")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv_strategy, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )

    # --- Run the Search ---
    grid_search.fit(X, y)

    # --- Display Best Results ---
    print("\n--- GridSearchCV Results ---")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    print("----------------------------")

    # --- Save the Best Model ---
    best_model = grid_search.best_estimator_
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model successfully trained and saved to '{MODEL_PATH}'")

def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        word_tokenize("test")
    except LookupError:
        print("Downloading NLTK 'punkt' package...")
        nltk.download('punkt_tab')
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK 'stopwords' package...")
        nltk.download('stopwords')
    try:
        lemmatizer.lemmatize("test")
    except LookupError:
        print("Downloading NLTK 'wordnet' package...")
        nltk.download('wordnet')


if __name__ == '__main__':
    download_nltk_data()
    train_and_evaluate()
