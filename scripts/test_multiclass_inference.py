import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from multiclass_intent_classifier import MulticlassIntentClassifier

def test_inference():
    print("Testing Multiclass Inference Engine...")
    clf = MulticlassIntentClassifier(model_path="models/intent_model_multiclass.pkl")
    
    test_cases = [
        # SQL Queries
        ("What is the total rainfall in Bihar in 2020?", "N/A"),
        ("Show me crop yields for Wheat in Punjab.", "N/A"),
        ("Did it rain in Bangalore yesterday?", "N/A"),
        
        # General Queries
        ("Hi, how are you?", "N/A"),
        ("Who is the Prime Minister of India?", "N/A"),
        ("Tell me a funny joke.", "N/A"),
        ("What is the capital of France?", "N/A"),

        # Platform Queries
        ("How do I log in to NPCYF?", "N/A"),
        ("What is the purpose of Data Management?", "N/A"),
        ("How to create a project in the platform?", "N/A"),
        ("Explain Feature Engineering module.", "N/A")
    ]
    
    correct = 0
    for query, expected in test_cases:
        result = clf.predict(query)
        predicted = result['intent']
        confidence = result['confidence']
        
        # Simple string comparison, assuming expected labels match model output labels
        is_correct = predicted == expected
        if is_correct:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        print(f"[{status}] Query: '{query}'")
        print(f"       Pred: {predicted} (Conf: {confidence:.2f}) | Exp: {expected}")
        print("-" * 50)
        
    #print(f"\nScore: {correct}/{len(test_cases)}")
    #if correct == len(test_cases):
        #print("ALL TESTS PASSED")
   ## else:
        #print("SOME TESTS FAILED")

if __name__ == "__main__":
    test_inference()
