import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Constants
DATA_PATH = "resourcesfortrainingtheintentclassifier/question_dataset_multiclass.csv"
MODEL_PATH = "models/intent_model_multiclass.pkl"

def train():
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    df['text'] = df['text'].astype(str)
    
    X = df['text']
    y = df['label'] # Labels are 'sql', 'general', 'platform'
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build Pipeline
    print("Training multiclass model...")
    # Logistic Regression with 'lbfgs' solver handles multinomial loss by default, which is good for multiclass
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=200)) # Increased max_iter for convergence
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print("Done!")

if __name__ == "__main__":
    train()
