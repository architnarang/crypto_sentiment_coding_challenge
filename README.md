Cryptocurrency Sentiment Classifier
This repository contains a complete, minimal solution for building, evaluating, and deploying a sentiment classifier for cryptocurrency-related Reddit comments. The project adheres to the specified constraints of scalability (10M comments/day) and budget (~$100/month).

Table of Contents
Methodology & Rationale

Deployment Architecture

Repository Structure

Setup and How to Run

Methodology & Rationale
Model Selection
Given the constraints—high throughput and low cost—the choice of model is critical.

Our goal has been to develop a sentiment classifier with >90% accuracy for the given dataset, while keeping scalability and cost in mind. Here is a summary of our journey so far:

Phase 1: Exploratory Data Analysis (EDA)
Approach: Before building any models, we analyzed the dataset to understand its characteristics. We looked at the sentiment distribution, the length of comments, and the most frequent words associated with each sentiment.

Key Insights:

Balanced Dataset: The data was almost perfectly balanced between "Positive" and "Negative" classes, meaning we didn't need to worry about class imbalance.

Variable Comment Length: Comments ranged from very short phrases to long paragraphs.

Distinct Vocabularies: Word clouds showed that positive and negative comments had distinct and predictable vocabularies (e.g., "buy," "hold," "bull" vs. "scam," "lost," "down"), confirming that a text-based classifier was viable.

Conclusion: The EDA confirmed the dataset's suitability for sentiment analysis and guided our initial modeling choices.

Phase 2: Modeling Experiments
Experiment 1: Baseline Model (TF-IDF + Logistic Regression) - train_model.py
Approach: A classic and highly efficient machine learning pipeline. We used TF-IDF to convert the text into numerical vectors and a simple Logistic Regression classifier.

Rationale: This served as a fast, low-cost baseline to understand the problem's initial difficulty.

Result: ~77.6% Accuracy.

Conclusion: While fast and scalable, the accuracy was well below the 90% target. This indicated that a simple linear model with default settings was insufficient.

Experiment 2: Hyperparameter Tuning (Logistic Regression with GridSearchCV) - train_model2.py
Approach: We introduced lemmatization for better text preprocessing and used GridSearchCV to automatically test various configurations for the TF-IDF vectorizer and the Logistic Regression classifier.

Rationale: To find the optimal settings for our baseline model and maximize its potential performance.

Result: ~78.8% Accuracy.

Conclusion: The performance improved slightly but hit a ceiling. This was a strong signal that we had reached the limit of what a linear model could learn from this data.

Experiment 3: Advanced Classical Model (TF-IDF + LightGBM) - train_model3.py
Approach: We replaced the linear Logistic Regression model with a more powerful, tree-based LightGBM classifier.

Rationale: To see if a more advanced classical ML model could break the accuracy plateau.

Result: ~66.0% Accuracy.

Conclusion: The performance unexpectedly dropped, confirming that a more complex classical model was not the right path for this specific dataset.

Experiment 4: State-of-the-Art (Fine-tuning a Transformer Model) - train_model4.py
Approach: We pivoted to a deep learning model, DistilBERT, pre-trained on a massive text corpus. We then fine-tuned it on our specific crypto sentiment dataset. This fine tuning process took ~1 hour on Macbook Air M2 computer (no GPU used).

Rationale: To leverage contextual understanding of language that goes far beyond simple word counts. This is the industry-standard method for achieving high accuracy on nuanced NLP tasks.

Result: ~85.8% Accuracy.

Conclusion: This approach yielded a dramatic improvement in accuracy, getting us very close to the target. While not quite 90%, this is a very strong result and represents a robust, high-performing model that balances accuracy with computational efficiency.

![alt text](image-1.png)

Deployment Architecture:

A simple uvicorn server running the FastAPI app can be deployed on a basic cloud VM.

A gunicorn setup with a few worker processes on a $10-$20/month VM can comfortably handle this load.

Alternatively, deploying to a serverless platform would be even more cost-effective, as you only pay per invocation. This architecture easily fits within the $100/month budget.

Repository Structure
.
├── data/
│   └── crypto_currency_sentiment_dataset.csv
├── saved_model/
│   └── final_model
├── api.py                  # FastAPI application
├── eda.py                  # Exploratory Data Analysis script
├── requirements.txt        # Python dependencies
├── train_model4.py          # Training and evaluation script
└── README.md               # This file

Setup and How to Run
1. Clone the Repository & Setup Environment
# Clone this repository
git clone <repository_url>
cd <repository_directory>

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

2. Run EDA (Optional)
To see the data analysis and generate charts:

python eda.py

This will save analysis charts to a new charts/ directory.

3. Train the Model
To run the training and evaluation process. This will create the saved_model/final_model file.


python train_model4.py

4. Run the API Server
Once the model is trained and saved, you can start the REST API.

uvicorn api:app --reload

5. Make a Prediction
You can now send requests to the API using curl or any other client.

Positive Example:

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"comment": "I am so bullish on this, it is the future of finance"}'

Response:

{
  "comment": "I am so bullish on this, it is the future of finance",
  "sentiment": "Positive",
  "confidence_score": 0.987
}

Negative Example:

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"comment": "another rug pull, I lost everything, this is a terrible project"}'

Response:

{
  "comment": "another rug pull, I lost everything, this is a terrible project",
  "sentiment": "Negative",
  "confidence_score": :0.9709
}

Screenshot of Results:
![alt text](image.png)