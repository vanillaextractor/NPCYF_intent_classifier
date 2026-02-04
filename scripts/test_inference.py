import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from intent_classifier import IntentClassifier

def test_inference():
    print("Testing Inference Engine...")
    clf = IntentClassifier(model_path="models/intent_model.pkl")
    
    test_cases = [
        ("What is the yield of Rice in Punjab in 2022?", "SQL"),
        ("Explain the concept of gradient descent", "General"),
        ("Show me rainfall data for Mumbai", "SQL"),
        ("How to normalize data?", "General"),
        ("Which state produced the most Sugar?", "SQL"),
        ("Best pesticides for cotton", "General"), # Ambiguous slightly? Should be General probably as it's not a direct statistic query from DB usually, but could be DB. Let's see model behavior. Dataset has "Best practices..." as General.
        ("Average temperature in Delhi", "SQL"),
        ("Who is the father of Green Revolution?", "General")
    ]
    
    correct = 0
    for query, expected in test_cases:
        result = clf.predict(query)
        predicted = result['intent']
        is_correct = predicted == expected
        if is_correct:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        print(f"[{status}] Query: '{query}' -> Pred: {predicted} ({result['confidence']:.2f}) | Exp: {expected}")
        
    print(f"\nScore: {correct}/{len(test_cases)}")
    if correct == len(test_cases):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

if __name__ == "__main__":
    test_inference()
