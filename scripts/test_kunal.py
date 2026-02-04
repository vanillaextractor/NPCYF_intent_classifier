import sys
import os
import csv
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from intent_classifier import IntentClassifier

# Constants
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/kunal_testset.csv')
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(__file__), '../kunal_evaluation_results.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/intent_model.pkl')

def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        clf = IntentClassifier(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading test data from {TEST_DATA_PATH}...")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data not found at {TEST_DATA_PATH}")
        return
        
    try:
        # Load dataset without header as inspection showed raw data starting line 1
        df = pd.read_csv(TEST_DATA_PATH, header=None, names=['text', 'label'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Evaluating {len(df)} queries...")
    
    results = []
    y_true = []
    y_pred = []
    
    # Map dataset labels to model labels
    # Dataset: 1 (SQL), 0 (General) - Inferred from inspection: 
    # "What is the total rainfall...?" -> 1
    # "What is the capital of India?" -> 0
    label_map = {
        1: "SQL",
        0: "General"
    }

    for index, row in df.iterrows():
        query = str(row['text'])
        original_label = row['label']
        try:
            expected_label = label_map.get(int(original_label), "Unknown")
        except ValueError:
            print(f"Warning: Invalid label '{original_label}' at row {index}. Skipping.")
            continue
            
        # Predict
        prediction = clf.predict(query)
        predicted_label = prediction['intent']
        confidence = prediction['confidence']
        
        is_correct = (predicted_label == expected_label)
        
        results.append({
            "query": query,
            "ground_truth_original": original_label,
            "ground_truth": expected_label,
            "predicted": predicted_label,
            "confidence": round(confidence, 4),
            "is_correct": is_correct
        })
        
        y_true.append(expected_label)
        y_pred.append(predicted_label)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['General', 'SQL'], output_dict=False)
    
    # Console Output
    print("\n" + "="*40)
    print("EVALUATION RESULTS (KUNAL TESTSET)")
    print("="*40)
    print(f"Total Queries: {len(df)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("-" * 40)
    print("Classification Report:")
    print(report)
    print("-" * 40)
    
    # Save Results to CSV
    print(f"Saving detailed results to {OUTPUT_CSV_PATH}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    evaluate()
