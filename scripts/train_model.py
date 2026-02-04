import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Constants
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/intent_model.pkl"

def train():
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing (Simple for now)
    df['text'] = df['text'].astype(str)
    
    X = df['text']
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build Pipeline
    print("Training model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(solver='liblinear', C=1.0))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['General', 'SQL']))
    
    # Save
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print("Done!")

if __name__ == "__main__":
    train()
