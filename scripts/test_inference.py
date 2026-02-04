import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from intent_classifier import IntentClassifier

def test_inference():
    print("Testing Inference Engine...")
    clf = IntentClassifier(model_path="models/intent_model.pkl")
    
    test_cases = [
       ("What is the total rainfall for the year 2010 in usa?","N/A")
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
        
    #print(f"\nScore: {correct}/{len(test_cases)}")
    #if correct == len(test_cases):
        #print("ALL TESTS PASSED")
    #else:
      #  print("SOME TESTS FAILED")

if __name__ == "__main__":
    test_inference()
