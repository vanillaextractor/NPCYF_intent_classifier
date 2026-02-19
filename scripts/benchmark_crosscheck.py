
import sys
import os
import time
import csv
import re
from interactive_inference import fix_generated_sql, get_db_credentials, execute_sql_query

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from multiclass_intent_classifier import MulticlassIntentClassifier

# Define the 33 specific questions
QUERIES = [
    "1. find the Maximum Temperature of kolkata in 2000",
    "2. ind the Maximum Temperature of kolkata in 2000",
    "3. what is the max rainfall in kolkata?",
    "4. in which year there was max rainfall in jodhpur?",
    "5. what is the maximum temerature in jodhpur?",
    "6. waht is the maximum temperature in delhi?",
    "7. what is the max temperature in chennai?",
    "8. what is the max temperature in delhi?",
    "9. list all the available districts in the database",
    "10. list all the state name from the district",
    "11. list all the state name from the database",
    "12. which state had highest rainfall in year 2020?",
    "13. which state had lowest rainfall?",
    "14. which state had lowest temperature ?",
    "15. which distric recored the lowest temperature?",
    "16. which distric recored the lowest temperature in 2020?",
    "17. is weekly rainfall data available in the database?",
    "18. is weekly rainfall data available in the database? this is sql query",
    "19. is weekly rainfall data is available yearwise?",
    "20. what is the weeekly rainfall data for each state in year 2020?",
    "21. what is the yield rate in rajathan in year 2020?",
    "22. find the Maximum Temperature of kolkata in 2000",
    "23. what is the max rainfall in kolkata?",
    "24. what is the maximum temperature ever recorded in jodhpur?",
    "25. what is the maximum temperature ever recorded in kolkata?",
    "26. what is the maximum temperature ever recorded in pune?",
    "27. what is the maximum temperature ever recorded in jodhpur?",
    "28. what is the maximum rainfall in jodhpur?",
    "29. in which year jodhpur highest rainfall?",
    "30. which district has highest rainfall in 2010?",
    "31. list all the districts available in the dataset?",
    "32. almorah tell me about average temperature , rainfall and all other feature of almora",
    "33. what is lowest temperature in almora?"
]

def clean_query_text(text):
    return re.sub(r'^\d+\.\s*', '', text).strip()

def run_benchmark():
    print("🚀 Starting Crosscheck Benchmark...")
    
    # 1. Load Models & DB Config
    try:
        clf = MulticlassIntentClassifier(model_path="models/intent_model_multiclass.pkl")
        print("✅ Intent Classifier Loaded.")

        from llama_cpp import Llama
        
        GENERAL_MODEL_PATH = "models/general_model.gguf"
        SQL_MODEL_PATH = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
        
        general_llm = None
        sql_llm = None
        
        if os.path.exists(GENERAL_MODEL_PATH):
            general_llm = Llama(model_path=GENERAL_MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)
            print("✅ General LLM Loaded.")
            
        if os.path.exists(SQL_MODEL_PATH):
            sql_llm = Llama(model_path=SQL_MODEL_PATH, n_gpu_layers=-1, n_ctx=8192, verbose=False)
            print("✅ SQL LLM Loaded.")
            
        schema_path = os.path.join(os.path.dirname(__file__), '../resourcesfortrainingtheintentclassifier/Database Schema.sql')
        with open(schema_path, 'r') as f:
            schema_content = f.read()
        print("✅ Schema Loaded.")
        
        print("🔐 Getting DB Credentials...")
        try:
             # Try getting from env, else defaults
             if os.environ.get("DB_HOST"):
                 db_config = get_db_credentials()
             else:
                 db_config = {
                    "host": "localhost",
                    "database": "postgres",
                    "user": "pulkitchauhan",
                    "password": None, 
                    "port": "5432"
                 }
                 print("⚠️ Using default DB credentials.")
        except:
             db_config = None
             
    except Exception as e:
        print(f"❌ Setup Failed: {e}")
        return

    # 2. Process Queries
    results = []
    
    for i, q_raw in enumerate(QUERIES):
        q_text = clean_query_text(q_raw)
        print(f"\n[{i+1}/{len(QUERIES)}] Processing: {q_text}")
        
        try:
            # Classification
            pred = clf.predict(q_text)
            intent = pred['intent']
            
            sql_query = "N/A"
            final_response = "N/A"
            error_msg = ""
            
            if intent == 'sql' and sql_llm:
                prompt = f"""<|im_start|>system
You are an expert SQL assistant. Your goal is to generate accurate SQL queries based EXACTLY on the provided schema.

CRITICAL SCHEMA RULES:
1. **LOCATION_TYPE**: Use `location_type = 'DISTRICT'` ONLY for fact tables. Use it in the JOIN condition.
2. **NO STATE DATA**: Data is NOT stored at the State level in fact tables. **ALWAYS** join through `DISTRICT` for State queries.
3. **JOIN RULES (CRITICAL)**:
   - Fact tables use `location_id`. They do NOT have `state_id`.
   - **Correct State Join Chain**: `JOIN npcyf_schema."DISTRICT" AS d ON fact.location_type = 'DISTRICT' AND fact.location_id = d.district_id JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id`
4. **AGGREGATION RULES**:
   - Rainfall -> `SUM(lr.rainfall_value)`.
   - Reservoir Level -> `MAX(lr.level)` for highest level, `SUM(lr.current_live_storage)` for storage.
5. **NO HALLUCINATION**: There is NO table named `LOCATION`. NEVER use a table named `LOCATION`.
6. DISTRICT QUERY PATTERN: Join DISTRICT table for district names.
7. Schema Prefix: ALWAYS use npcyf_schema prefix.
8. String Matching: Use ILIKE for names.
9. Date Handling: Use EXTRACT.
10. Aggregation: Group by all non-aggregated columns.

EXAMPLES:
- User: "Which state has highest rainfall in 2010?"
  Assistant: ```sql
SELECT s.state_name
FROM npcyf_schema."LOCATION_RAINFALL" AS lr
JOIN npcyf_schema."DISTRICT" AS d ON lr.location_type = 'DISTRICT' AND lr.location_id = d.district_id
JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id
WHERE EXTRACT(YEAR FROM lr.rainfall_recorded_date) = 2010
GROUP BY s.state_name
ORDER BY SUM(lr.rainfall_value) DESC
LIMIT 1;
```
- User: "Which state has the highest reservoir level in 2010?"
  Assistant: ```sql
SELECT s.state_name
FROM npcyf_schema."LOCATION_RESERVOIR_LEVEL" AS lr
JOIN npcyf_schema."DISTRICT" AS d ON lr.location_type = 'DISTRICT' AND lr.location_id = d.district_id
JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id
WHERE EXTRACT(YEAR FROM lr.reservoir_level_recorded_date) = 2010
GROUP BY s.state_name
ORDER BY MAX(lr.level) DESC
LIMIT 1;
```

Schema Definition:
{schema_content}
<|im_end|>
<|im_start|>user
{q_text}
<|im_end|>
<|im_start|>assistant
```sql
"""
                output = sql_llm(prompt, max_tokens=256, stop=["```", "<|im_end|>"], echo=False)
                raw_sql = output['choices'][0]['text'].strip()
                sql_query = raw_sql # Store raw logic before fix/exec
                
                # Execute SQL
                try:
                    fixed_sql = fix_generated_sql(raw_sql)
                    if fixed_sql != raw_sql:
                        sql_query = fixed_sql # Update stored SQL to what was actually run
                    
                    # Capture stdout to see results? or modify execute_sql_query (not modifying helper currently)
                    # We will assume execute_sql_query returns (cols, rows) based on previous analysis
                    columns, rows = execute_sql_query(fixed_sql, db_config)
                    
                    if rows is not None:
                         if len(rows) > 0:
                             final_response = f"Result: {rows}"
                         else:
                             final_response = "No results found."
                    else:
                        final_response = "Query executed successfully."
                        
                except Exception as db_err:
                    error_msg = str(db_err)
                    final_response = "Database Error"

            elif intent == 'general' and general_llm:
                output = general_llm(f"User: {q_text}\nAssistant:", max_tokens=256, stop=["User:", "\n"], echo=False)
                final_response = output['choices'][0]['text'].strip()
                
            elif intent == 'platform':
                final_response = "Platform query detected. Refer to documentation."
            
            # Append Result
            results.append({
                "ID": i+1,
                "Query": q_text,
                "Intent": intent,
                "SQL": sql_query,
                "Response": final_response,
                "Error": error_msg
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "ID": i+1, "Query": q_text, "Intent": "ERROR", 
                "SQL": "", "Response": "", "Error": str(e)
            })

    # 3. Write CSV
    csv_path = "stresstest_new.csv"
    headers = ["ID", "Query", "Intent", "SQL", "Response", "Error"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n✅ Benchmark Complete! Saved to {csv_path}")

if __name__ == "__main__":
    run_benchmark()
