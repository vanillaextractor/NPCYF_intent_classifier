import pickle
import os
import numpy as np

class MulticlassIntentClassifier:
    def __init__(self, model_path="models/intent_model_multiclass.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, query):
        if not self.model:
            raise RuntimeError("Model is not loaded.")
        
        # Predict
        prediction = self.model.predict([query])[0]
        
        # Get probabilities
        probs = self.model.predict_proba([query])[0]
        confidence = max(probs)
        
        # Get class mapping
        classes = self.model.classes_
        class_index = np.where(classes == prediction)[0][0]
        
        return {
            "query": query,
            "intent": prediction,
            "confidence": float(confidence),
            "intent_id": int(class_index),
            "probabilities": {cls: float(prob) for cls, prob in zip(classes, probs)}
        }

if __name__ == "__main__":
    # Test
    clf = MulticlassIntentClassifier()
    test_queries = [
        "What is the yield of wheat?", 
        "Tell me a joke.", 
        "How do I create a dataset in NPCYF?",
        "Show me rainfall data for Bihar."
    ]
    
    for q in test_queries:
        print(f"Query: {q}")
        print(f"Result: {clf.predict(q)}")
        print("-" * 30)
