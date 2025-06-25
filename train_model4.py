import os
import torch
import joblib
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_FILE = os.path.join('data', 'crypto_currency_sentiment_dataset.csv')
MODEL_DIR = 'saved_model'
MODEL_NAME = "distilbert-base-uncased" # A lightweight and effective transformer

def compute_metrics(p):
    """Computes and returns metrics for evaluation."""
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_and_evaluate():
    """
    Loads data, fine-tunes a DistilBERT model, and saves the result.
    This approach provides much higher accuracy than traditional methods.
    """
    # --- 1. Load and Prepare Data ---
    try:
        df = pd.read_csv(DATA_FILE)
        df.dropna(subset=['Comment', 'Sentiment'], inplace=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE}'. Please check the path.")
        return

    # Create labels: 1 for Positive, 0 for Negative
    df['label'] = df['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
    # Rename 'Comment' to 'text' for consistency
    df = df.rename(columns={'Comment': 'text'})

    # --- 2. Create Train/Test Split ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Convert pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # --- 3. Tokenize the Data ---
    print(f"Loading tokenizer for model: '{MODEL_NAME}'")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # --- 4. Load Pre-trained Model ---
    print(f"Loading pre-trained model: '{MODEL_NAME}'")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # --- 5. Set up Training with Optimizations for M2 Mac ---
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "training_output"),
        num_train_epochs=3,
        per_device_train_batch_size=8,          # REDUCED BATCH SIZE to lower memory usage
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,          # ADDED to compensate for smaller batch size
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_pin_memory=False             # ADDED to address MPS warning and improve stability
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
    )

    # --- 6. Train the Model ---
    print("\n--- Starting Model Fine-Tuning ---")
    trainer.train()
    print("--- Fine-Tuning Complete ---")

    # --- 7. Evaluate the Best Model ---
    print("\n--- Evaluating Final Model on Test Set ---")
    eval_results = trainer.evaluate()
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1-Score: {eval_results['eval_f1']:.4f}")
    print("------------------------------------------")

    # --- 8. Save the Final Model and Tokenizer ---
    final_model_path = os.path.join(MODEL_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nBest model and tokenizer saved to '{final_model_path}'")

if __name__ == '__main__':
    # Set a seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_and_evaluate()
