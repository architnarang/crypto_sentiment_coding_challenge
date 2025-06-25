import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_DIR = 'saved_model'
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model")

# --- API Initialization ---
app = FastAPI(
    title="High-Accuracy Cryptocurrency Sentiment Classifier API",
    description="An API using a fine-tuned Transformer model to classify sentiment.",
    version="2.0.0"
)

# --- Load Fine-Tuned Model ---
sentiment_classifier = None
try:
    if not os.path.exists(FINAL_MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found at {FINAL_MODEL_PATH}")

    # Load the model and tokenizer from the saved path
    model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH)
    
    # Create a sentiment analysis pipeline
    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer
    )
    print(f"Fine-tuned model loaded successfully from {FINAL_MODEL_PATH}")

except Exception as e:
    print(f"FATAL ERROR: Could not load the model. {e}")
    print("Please run train_model.py to fine-tune and save the model first.")


# --- Pydantic Models for Request and Response ---
class CommentRequest(BaseModel):
    comment: str

class PredictionResponse(BaseModel):
    comment: str
    sentiment: str
    confidence_score: float

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    if sentiment_classifier is None:
        raise RuntimeError("Failed to load the sentiment analysis model. API cannot start.")

@app.get("/", summary="Root endpoint to check API status")
async def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "High-Accuracy Sentiment Classifier API is running."}

@app.post("/predict", response_model=PredictionResponse, summary="Predict sentiment of a comment")
async def predict_sentiment(request: CommentRequest):
    """
    Predicts the sentiment of a given comment using a fine-tuned Transformer model.
    """
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment text cannot be empty.")
    
    try:
        # The pipeline handles tokenization and prediction
        results = sentiment_classifier(request.comment)
        
        # The output from pipeline is a list of dicts
        # e.g., [{'label': 'LABEL_1', 'score': 0.99}]
        prediction = results[0]
        
        # Convert label back to 'Positive' or 'Negative'
        # The model was trained with Positive=1, Negative=0
        sentiment = "Positive" if prediction['label'] == 'LABEL_1' else "Negative"
        confidence = prediction['score']
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    return PredictionResponse(
        comment=request.comment,
        sentiment=sentiment,
        confidence_score=round(confidence, 4)
    )

