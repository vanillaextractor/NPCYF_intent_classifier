import pickle
import os

class IntentClassifier:
    def __init__(self, model_path="models/intent_model.pkl"):
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
        
        # Determine intent
        # 0: General, 1: SQL
        prediction = self.model.predict([query])[0]
        label = "SQL" if prediction == 1 else "General"
        
        # Get probabilities
        probs = self.model.predict_proba([query])[0]
        confidence = max(probs)
        
        return {
            "query": query,
            "intent": label,
            "confidence": float(confidence),
            "intent_id": int(prediction)
        }

if __name__ == "__main__":
    # Quick test
    clf = IntentClassifier()
    print(clf.predict("Yield of wheat in 2020"))
    print(clf.predict("What is deep learning?"))
