import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from multiclass_intent_classifier import MulticlassIntentClassifier

def interactive_mode():
    # Initialize LLMs
    general_llm = None
    sql_llm = None
    
    try:
        from llama_cpp import Llama
        
        # Paths - Update these or place models here
        GENERAL_MODEL_PATH = "models/general_model.gguf"
        SQL_MODEL_PATH = "models/sql_model.gguf"
        
        if os.path.exists(GENERAL_MODEL_PATH):
            print(f"Loading General Model from {GENERAL_MODEL_PATH}...")
            general_llm = Llama(model_path=GENERAL_MODEL_PATH, verbose=False)
        else:
            print(f"Warning: General model not found at {GENERAL_MODEL_PATH}. General queries will only show intent.")
            
        if os.path.exists(SQL_MODEL_PATH):
            print(f"Loading SQL Model from {SQL_MODEL_PATH}...")
            sql_llm = Llama(model_path=SQL_MODEL_PATH, verbose=False)
        else:
            print(f"Warning: SQL model not found at {SQL_MODEL_PATH}. SQL queries will only show intent.")
            
    except ImportError:
        print("Warning: llama-cpp-python not installed. LLM generation disabled.")
    except Exception as e:
        print(f"Error initializing LLMs: {e}")

    try:
        clf = MulticlassIntentClassifier(model_path="models/intent_model_multiclass.pkl")
        print("Intent Classifier loaded successfully!")
        print("-" * 50)
        print("Enter a query to detect its intent and get an LLM response.")
        print("Type 'exit' or 'quit' to stop.")
        print("-" * 50)
    except Exception as e:
        print(f"Error loading classifier model: {e}")
        return

    while True:
        try:
            user_input = input("\n📝 Enter Query: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting... Goodbye!")
                break
            
            if not user_input:
                continue
            
            start_time = time.time()
                
            result = clf.predict(user_input)
            
            intent = result['intent']
            confidence = result['confidence']
            
            print(f"🤖 Prediction: {intent.upper()}")
            print(f"📊 Confidence: {confidence:.2%}")
            
            # Routing Logic
            if intent == "general":
                if general_llm:
                    print(f"🧠 Generating response from General LLM...")
                    output = general_llm(
                        f"User: {user_input}\nAssistant:", 
                        max_tokens=128, 
                        stop=["User:", "\n"], 
                        echo=False
                    )
                    print(f"💬 Response: {output['choices'][0]['text'].strip()}")
                else:
                    print("⚠️  General LLM not loaded.")
            
            elif intent == "sql":
                if sql_llm:
                    print(f"🧠 Generating SQL from Text2SQL LLM...")
                    # Simple prompt for Text2SQL
                    prompt = f"Convert this text to SQL: {user_input}\nSQL:"
                    output = sql_llm(
                        prompt, 
                        max_tokens=128, 
                        stop=["\n", ";"], 
                        echo=False
                    )
                    print(f"💾 SQL: {output['choices'][0]['text'].strip()}")
                else:
                    print("⚠️  SQL LLM not loaded.")
            
            elif intent == "platform":
                print("ℹ️  Platform query detected. Refer to documentation.")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"⏱️  Time Taken: {elapsed_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\nExiting... Goodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    interactive_mode()
