import sys
import os

try:
    from llama_cpp import Llama
    
    SQL_MODEL_PATH = "models/sql_model.gguf"
    
    if not os.path.exists(SQL_MODEL_PATH):
        print(f"Error: SQL model not found at {SQL_MODEL_PATH}")
        sys.exit(1)
        
    print(f"Loading SQL Model from {SQL_MODEL_PATH}...")
    sql_llm = Llama(model_path=SQL_MODEL_PATH, verbose=False)
    
    test_query = "what is the annual rainfall in bihar?"
    
    print("-" * 50)
    print(f"Test Query: {test_query}")
    print("-" * 50)

    # Test 1: Original Prompt
    print("Test 1: Original Prompt")
    prompt1 = f"Convert this text to SQL: {test_query}\nSQL:"
    output1 = sql_llm(
        prompt1, 
        max_tokens=128, 
        stop=["\n", ";"], 
        echo=False
    )
    print(f"Output: '{output1['choices'][0]['text']}'")
    
    # Test 2: Standard Chat Prompt (if it's an instruct model)
    print("\nTest 2: Chat Prompt")
    prompt2 = f"<|im_start|>user\nConvert this text to SQL: {test_query}<|im_end|>\n<|im_start|>assistant\nSELECT"
    output2 = sql_llm(
        prompt2, 
        max_tokens=128, 
        stop=["<|im_end|>", ";"], 
        echo=False
    )
    print(f"Output: SELECT'{output2['choices'][0]['text']}'")

    # Test 3: Simple Prompt without newline stop
    print("\nTest 3: Simple Prompt (No \\n stop)")
    prompt3 = f"Provide SQL for: {test_query}\nSQL:"
    output3 = sql_llm(
        prompt3, 
        max_tokens=128, 
        stop=[";"], 
        echo=False
    )
    print(f"Output: '{output3['choices'][0]['text']}'")

except ImportError:
    print("llama-cpp-python not installed.")
except Exception as e:
    print(f"Error: {e}")
