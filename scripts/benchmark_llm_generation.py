import os
import sys
import json
import time
import re
import csv
import psycopg2
from dotenv import load_dotenv

# Add src to path for intent classifier
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from multiclass_intent_classifier import MulticlassIntentClassifier

# Load environment variables
load_dotenv()

def fix_generated_sql(sql):
    """
    Robustly fixes common SQL generation errors, specifically missing schema prefixes.
    Ignores keys that are inside single quotes (string literals).
    """
    tables = [
        "CROP", "SEASON", "STATE", "DISTRICT", 
        "LOCATION_RAINFALL", "LOCATION_TEMPERATURE", "LOCATION_RESERVOIR_LEVEL",
        "RAINFALL_DATA_COLLECTION", "TEMPERATURE_DATA_COLLECTION", "RESERVOIR_LEVEL_DATA_COLLECTION",
        "FEATURE_MASTER", "TRAINING_FEATURE_SET", "TRAINING_FEATURE_SET_FEATURE",
        "TRAINING_FEATURE_SET_DATA", "TRAINING_FEATURE_SET_DATA_INDIVIDUAL_FEATURE",
        "TRAINING_FEATURE_SET_DATA_SOURCE_COLLECTION", "TRAINING_FEATURE_SET_FEATURE_TEMPORAL_INTERVAL_ITEM",
        "TRAINING_DATASET_COLLECTION", "TRAINING_DATASET_COLLECTION_ITEM",
        "MODEL_BUILD_ALGORITHM", "MODEL_BUILD_ALGORITHM_HYPERPARAMETER",
        "MODEL_BUILD_CONFIG_COLLECTION", "MODEL_BUILD_CONFIG", "MODEL_BUILD_CONFIG_COLLECTION_ITEM",
        "MODEL_BUILD_CONFIG_ITEM", "MODEL_BUILD_BATCH_RUN", "MODEL_BUILD_RUN",
        "MODEL_BUILD_RUN_ARTIFACT", "MODEL_BUILD_RUN_METRIC",
        "FORECAST_CONFIG_COLLECTION", "FORECAST_EXOGENEOUS_DATA_ESTIMATION_CONFIG_COLLECTION",
        "FORECAST_BATCH", "FORECAST_BATCH_RUN", "FORECAST_CONFIG", "FORECAST_RUN", "FORECAST_RUN_RESULT"
    ]
    parts = sql.split("'")
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        for table in tables:
            pattern_unquoted = re.compile(f'npcyf_schema\.{table}\\b(?!\")', re.IGNORECASE) 
            chunk = pattern_unquoted.sub(f'npcyf_schema."{table}"', chunk)
            pattern_no_prefix = re.compile(f'(?<!npcyf_schema\.)(?<!npcyf_schema\.")\\b{table}\\b', re.IGNORECASE)
            chunk = pattern_no_prefix.sub(f'npcyf_schema."{table}"', chunk)
        parts[i] = chunk
    return "'".join(parts)

def parse_testing_json(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    all_cases = []
    blocks = re.findall(r'\[.*?\]', content, re.DOTALL)
    for block in blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                all_cases.extend(data)
        except json.JSONDecodeError:
            continue
    return all_cases

def run_llm_benchmark():
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    input_file = os.path.join(os.path.dirname(__file__), '../data/testing.json')
    output_file = os.path.join(os.path.dirname(__file__), '../data/benchmark_llm_results.csv')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Loading LLMs and Classifier...")
    from llama_cpp import Llama
    SQL_MODEL_PATH = config['models']['sql']['path']
    sql_llm = Llama(
        model_path=SQL_MODEL_PATH, 
        n_gpu_layers=config['models']['sql']['n_gpu_layers'],
        n_ctx=config['models']['sql']['n_ctx'],
        verbose=False
    )
    
    clf_path = config['models']['intent_classifier']['path']
    clf = MulticlassIntentClassifier(model_path=clf_path)
    
    schema_path = os.path.join(os.path.dirname(__file__), '..', config['paths']['schema'])
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "port": os.getenv("DB_PORT", "5432")
    }
    
    test_cases = parse_testing_json(input_file)
    print(f"Loaded {len(test_cases)} test cases.")
    
    # Run only a subset for verification if needed
    LIMIT = 10 # Change to None for full run
    if LIMIT:
        test_cases = test_cases[:LIMIT]
        print(f"Limiting benchmark to first {LIMIT} cases.")

    results = []
    
    INDIAN_STATES = [
        "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", 
        "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", 
        "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur", 
        "meghalaya", "mizoram", "nagaland", "odisha", "punjab", 
        "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", 
        "uttar pradesh", "uttarakhand", "west bengal", "delhi", 
        "jammu and kashmir", "puducherry", "chandigarh", "ladakh"
    ]

    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        for i, case in enumerate(test_cases):
            question = case.get("question", "N/A")
            gold_sql = case.get("sql") or case.get("query", "")
            
            print(f"[{i+1}/{len(test_cases)}] Processing: {question[:50]}...")
            
            # 1. Intent Classification
            clf_result = clf.predict(question)
            intent = clf_result['intent']
            
            if intent != "sql":
                print(f"   Skipping: Non-SQL intent ({intent})")
                results.append({
                    "question": question,
                    "gold_sql": gold_sql,
                    "pred_sql": "N/A (Intent: " + intent + ")",
                    "status": "Skipped"
                })
                continue
            
            # 2. Location Hint
            detected_state = None
            lower_input = question.lower()
            for state in INDIAN_STATES:
                if state in lower_input:
                    detected_state = state
                    break
            
            if detected_state:
                 location_hint = f"LOCATION: '{detected_state.title()}' (STATE). FILTER COLUMN: `s.state_name`. Data can be `location_type='STATE'` OR `location_type='DISTRICT'` (summed/aggregated)."
            else:
                 location_hint = "LOCATION: City/District. FILTER COLUMN: `d.district_name`. NEVER filter by `state_name`. YOU MUST JOIN `DISTRICT` table."

            # 3. SQL Generation
            prompt = f"""<|im_start|>system
You are an expert SQL assistant. Your goal is to generate accurate SQL queries based EXACTLY on the provided schema.

CRITICAL INSTRUCTION:
{location_hint}

RULES:
1. **LOCATION_TYPE**: Use `location_type = 'DISTRICT'` ONLY for fact tables. Use it in the JOIN condition.
2. **NO STATE DATA**: Data is NOT stored at the State level in fact tables. **ALWAYS** join through `DISTRICT` for State queries.
3. **JOIN RULES (CRITICAL)**:
   - Fact tables use `location_id`. They do NOT have `state_id`.
   - **Correct State Join Chain**: `JOIN npcyf_schema."DISTRICT" AS d ON fact.location_type = 'DISTRICT' AND fact.location_id = d.district_id JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id`
4. **AGGREGATION RULES**:
   - Rainfall -> `SUM(lr.rainfall_value)`.
   - Reservoir Level -> `MAX(lr.level)` for highest level, `SUM(lr.current_live_storage)` for storage.
5. **NO HALLUCINATION**: There is NO table named `LOCATION`. NEVER use a table named `LOCATION`.
6. **DATA SOURCE**: ALL data comes from FACT tables (`LOCATION_RAINFALL`, `LOCATION_TEMPERATURE`).
7. **NO SHORTCUTS**: Join `DISTRICT` or `STATE` for names.
8. **PREFIX**: Use `npcyf_schema` prefix.
9. **STRING MATCHING**: ALWAYS use `ILIKE` with `%` wildcards for names.
10. **DATE**: Use `EXTRACT(YEAR/MONTH FROM date_col)`.

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

JOIN PATTERNS:

**IF DISTRICT (Hint says DISTRICT):**
*Example: Total Rainfall in Jodhpur (Aggregation on District)*
```sql
SELECT SUM(lr.rainfall_value)
FROM npcyf_schema."LOCATION_RAINFALL" AS lr
JOIN npcyf_schema."DISTRICT" AS d ON lr.location_type = 'DISTRICT' AND lr.location_id = d.district_id
JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id
WHERE d.district_name ILIKE '%Jodhpur%'
```

**IF STATE (Hint says STATE):**
*Option 1: Direct State Data*
```sql
FROM npcyf_schema."LOCATION_TEMPERATURE" AS lt
JOIN npcyf_schema."STATE" AS s ON lt.location_type = 'STATE' AND lt.location_id = s.state_id
WHERE s.state_name ILIKE '%Rajasthan%'
```
*Option 2: Aggregated via Districts (Use for Totals/Sums)*
```sql
FROM npcyf_schema."LOCATION_TEMPERATURE" AS lt
JOIN npcyf_schema."DISTRICT" AS d ON lt.location_type = 'DISTRICT' AND lt.location_id = d.district_id
JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id
WHERE s.state_name ILIKE '%Bihar%'
```

Schema Definition:
{schema_content}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
```sql
"""
            output = sql_llm(
                prompt, 
                max_tokens=256, 
                stop=["```", "<|im_end|>"], 
                echo=False,
                temperature=0.1
            )
            pred_sql_raw = output['choices'][0]['text'].strip()
            pred_sql = fix_generated_sql(pred_sql_raw)
            
            # 4. Execution and Comparison
            gold_count = -1
            pred_count = -1
            error_msg = ""
            
            try:
                cur.execute(gold_sql)
                gold_count = len(cur.fetchall()) if cur.description else 0
            except Exception as e:
                print(f"   Gold SQL Error: {e}")
                conn.rollback()
            
            try:
                cur.execute(pred_sql)
                pred_count = len(cur.fetchall()) if cur.description else 0
            except Exception as e:
                print(f"   Pred SQL Error: {e}")
                error_msg = str(e)
                conn.rollback()
            
            status = "Match" if gold_count == pred_count and gold_count != -1 else "Mismatch"
            if error_msg:
                status = f"Error: {error_msg}"
            
            print(f"   Status: {status} (Gold: {gold_count}, Pred: {pred_count})")
            
            results.append({
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "gold_rows": gold_count,
                "pred_rows": pred_count,
                "status": status
            })
            
        conn.close()
    except Exception as e:
        print(f"Fatal Error: {e}")

    # Write Results
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "gold_sql", "pred_sql", "gold_rows", "pred_rows", "status"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Benchmark complete. Results saved to {output_file}")

if __name__ == "__main__":
    run_llm_benchmark()
