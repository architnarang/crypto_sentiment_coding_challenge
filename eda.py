import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# --- Configuration ---
DATA_FILE = os.path.join('data', 'crypto_currency_sentiment_dataset.csv')
CHARTS_DIR = 'charts'

def perform_eda():
    """
    Loads the dataset and performs exploratory data analysis.
    Generates and saves several plots to understand the data.
    """
    # --- Load Data ---
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE}'.")
        print("Please download the dataset and place it in the 'data/' directory.")
        return

    print("--- Dataset Head ---")
    print(df.head())
    print("\n--- Dataset Info ---")
    df.info()

    # --- Check for Missing Values ---
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    # Drop rows with missing comments if any
    df.dropna(subset=['Comment'], inplace=True)
    df['Sentiment'] = df['Sentiment'].astype('category')


    # --- 1. Sentiment Distribution ---
    print("\n--- Sentiment Distribution ---")
    sentiment_counts = df['Sentiment'].value_counts()
    print(sentiment_counts)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.title('Sentiment Distribution in Comments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    save_plot('sentiment_distribution.png')
    plt.show()


    # --- 2. Comment Length Analysis ---
    df['comment_length'] = df['Comment'].apply(len)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='comment_length', hue='Sentiment', multiple='stack', bins=50, palette='magma')
    plt.title('Distribution of Comment Length by Sentiment')
    plt.xlabel('Comment Character Length')
    plt.ylabel('Frequency')
    save_plot('comment_length_distribution.png')
    plt.show()


    # --- 3. Word Cloud Generation ---
    generate_word_clouds(df)

    print("\nEDA complete. Charts saved to 'charts/' directory.")


def generate_word_clouds(df):
    """Generates and saves word clouds for positive and negative sentiments."""
    positive_comments = ' '.join(df[df['Sentiment'] == 'Positive']['Comment'])
    negative_comments = ' '.join(df[df['Sentiment'] == 'Negative']['Comment'])

    # Positive Word Cloud
    plt.figure(figsize=(10, 7))
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_comments)
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Positive Comments')
    save_plot('wordcloud_positive.png')
    plt.show()

    # Negative Word Cloud
    plt.figure(figsize=(10, 7))
    wordcloud_neg = WordCloud(width=800, height=400, background_color='black').generate(negative_comments)
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Negative Comments')
    save_plot('wordcloud_negative.png')
    plt.show()


def save_plot(filename):
    """Saves the current matplotlib plot to the charts directory."""
    if not os.path.exists(CHARTS_DIR):
        os.makedirs(CHARTS_DIR)
    plt.savefig(os.path.join(CHARTS_DIR, filename), bbox_inches='tight')


if __name__ == '__main__':
    perform_eda()
