import os
import sys
import json
import time
import re
import csv
import psycopg2
from dotenv import load_dotenv

# Add src to path for intent classifier dependencies
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

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
            pattern_unquoted = re.compile(rf'npcyf_schema\.{table}\b(?!\")', re.IGNORECASE) 
            chunk = pattern_unquoted.sub(f'npcyf_schema."{table}"', chunk)
            pattern_no_prefix = re.compile(rf'(?<!npcyf_schema\.)(?<!npcyf_schema\.")\b{table}\b', re.IGNORECASE)
            chunk = pattern_no_prefix.sub(f'npcyf_schema."{table}"', chunk)
        parts[i] = chunk
    return "'".join(parts)

def fetch_all_states(db_config):
    conn = None
    states = []
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute('SELECT "state_name" FROM npcyf_schema."STATE"')
        rows = cur.fetchall()
        states = [row[0].lower() for row in rows]
    except Exception as e:
        print(f"Warning: Could not fetch states: {e}")
    finally:
        if conn: conn.close()
    return states

def fetch_all_districts(db_config):
    conn = None
    districts = []
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute('SELECT "district_name" FROM npcyf_schema."DISTRICT"')
        rows = cur.fetchall()
        districts = [row[0].lower() for row in rows]
    except Exception as e:
        print(f"Warning: Could not fetch districts: {e}")
    finally:
        if conn: conn.close()
    return districts

def get_pruned_schema(query_type, full_schema_str):
    """
    Simplistic pruning for the benchmarking script.
    """
    if query_type == "DISTRICT":
        return """- DISTRICT: district_id, district_name, state_id
- LOCATION_TEMPERATURE: temperature_value, temperature_recorded_date, temperature_recorded_year
- LOCATION_RAINFALL: rainfall_value, rainfall_recorded_date, rainfall_recorded_year
- LOCATION_RESERVOIR_LEVEL: level, frl, current_live_storage, reservoir_level_recorded_date"""
    elif query_type == "STATE":
        # For STATE, we NEED both DISTRICT and STATE tables
        return """- STATE: state_id, state_name
- DISTRICT: district_id, district_name, state_id
- LOCATION_TEMPERATURE: temperature_value, temperature_recorded_date, temperature_recorded_year
- LOCATION_RAINFALL: rainfall_value, rainfall_recorded_date, rainfall_recorded_year
- LOCATION_RESERVOIR_LEVEL: level, frl, current_live_storage, reservoir_level_recorded_date"""
    else:
        # GENERAL Mode - Full relevant schema
        return """- STATE: state_id, state_name
- DISTRICT: district_id, district_name, state_id
- SEASON: season_id, season_name
- CROP: crop_id, crop_name
- LOCATION_TEMPERATURE: temperature_value, temperature_recorded_date, temperature_recorded_year
- LOCATION_RAINFALL: rainfall_value, rainfall_recorded_date, rainfall_recorded_year
- LOCATION_RESERVOIR_LEVEL: level, frl, current_live_storage, reservoir_level_recorded_date"""

DISTRICT_PROMPT_TEMPLATE = """<|im_start|>system
You are a SQL assistant specialized in DISTRICT-level queries.
Your goal is to generate accurate SQL for the specified DISTRICT using the provided schema.

STRICT RULES:
1. **LOCATION_TYPE**: ALWAYS use `location_type = 'DISTRICT'` in the JOIN condition.
2. **JOIN DISTRICT**: ALWAYS join `npcyf_schema."DISTRICT"` table to filter by `district_name`.
3. **ONLY DISTRICT**: Use ONLY `d.district_name`. NEVER use `state_name` or any state-level filtering.
4. **FACT DATA**: Data comes from `LOCATION_RAINFALL`, `LOCATION_TEMPERATURE`, or `LOCATION_RESERVOIR_LEVEL`.
5. **NO HALLUCINATION**: There is NO table named `LOCATION`. NEVER use a table named `LOCATION`.
6. **MATCHING**: Use `ILIKE` with `%` wildcards for name matching.
7. **DATE**: Use `EXTRACT(YEAR/MONTH FROM date_col)`. DO NOT use `LIKE` on DATE columns.

SCHEMA:
{pruned_schema}

Examples for {example_location_name}:
*Rainfall*:
```sql
SELECT SUM(lr.rainfall_value)
FROM npcyf_schema."LOCATION_RAINFALL" AS lr
JOIN npcyf_schema."DISTRICT" AS d ON lr.location_type = 'DISTRICT' AND lr.location_id = d.district_id
WHERE d.district_name ILIKE '%{example_location_name}%'
```
<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
```sql
"""

STATE_PROMPT_TEMPLATE = """<|im_start|>system
You are a SQL assistant specialized in STATE-level queries.
Your goal is to generate accurate SQL for the specified STATE using the provided schema.

STRICT RULES:
1. **LOCATION_TYPE**: ALWAYS use `location_type = 'DISTRICT'` or `location_type = 'STATE'` in the JOIN condition.
2. **AGGREGATION**: Data is ONLY stored at the DISTRICT level. ALWAYS join fact tables to `DISTRICT` table, then join `DISTRICT` to `STATE` to filter by `state_name`.
3. **ONLY STATE**: Use ONLY `s.state_name` for filtering.
4. **FACT DATA**: Data comes from `LOCATION_RAINFALL`, `LOCATION_TEMPERATURE`, or `LOCATION_RESERVOIR_LEVEL`.
5. **NO HALLUCINATION**: There is NO table named `LOCATION`. NEVER use a table named `LOCATION`.
6. **MATCHING**: Use `ILIKE` with `%` wildcards for name matching.
7. **DATE**: Use `EXTRACT(YEAR/MONTH FROM date_col)`. DO NOT use `LIKE` on DATE columns.

SCHEMA:
{pruned_schema}

Examples for {example_location_name}:
*State Rainfall (Aggregated)*:
```sql
SELECT SUM(lr.rainfall_value)
FROM npcyf_schema."LOCATION_RAINFALL" AS lr
JOIN npcyf_schema."DISTRICT" AS d ON lr.location_type = 'DISTRICT' AND lr.location_id = d.district_id
JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id
WHERE s.state_name ILIKE '%{example_location_name}%'
```
<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
```sql
"""

GENERAL_SQL_PROMPT_TEMPLATE = """<|im_start|>system
You are a SQL assistant.
Your goal is to generate accurate SQL using the provided schema.

STRICT RULES:
1. **LOCATION_TYPE**: Use `location_type = 'DISTRICT'` ONLY for fact tables. Use it in the JOIN condition.
2. **NO STATE RAINFALL/RESERVOIR**: Data is NOT stored at the State level in fact tables. **ALWAYS** join through `DISTRICT` for State queries.
3. **JOIN RULES (CRITICAL)**:
   - Fact tables use `location_id`. They do NOT have `state_id`.
   - **Correct State Join Chain**: `JOIN npcyf_schema."DISTRICT" AS d ON fact.location_type = 'DISTRICT' AND fact.location_id = d.district_id JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id`
4. **AGGREGATION RULES**:
   - For **Rainfall**: Use `SUM(lr.rainfall_value)`.
   - For **Reservoir Level**: Use `MAX(lr.level)` for "highest level" and `SUM(lr.current_live_storage)` for storage.
5. **NO HALLUCINATION**: There is NO table named `LOCATION`. NEVER use a table named `LOCATION`.
6. **DATE**: Use `EXTRACT(YEAR/MONTH FROM date_col)`.
7. **MATCHING**: Use `ILIKE` with `%` for flexible matching of names.

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

SCHEMA:
{pruned_schema}
<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
```sql
"""

def run_benchmark():
    # Fixed paths as per user request
    base_dir = "/Users/pulkitchauhan/Desktop/IDEAS_TIH/intent_clasifier/benchmark"
    input_csv = os.path.join(base_dir, "input.csv")
    output_csv = os.path.join(base_dir, "output.csv")
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found at {input_csv}")
        return

    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Mapping to User's CSV structure
    fieldnames = ["sql_queries", "gold_value", "model_answer", "match", "gold_sql", "model_sql", "timing(in secs)", "Mode", "Detected_Entity"]

    print("Loading Qwen SQL Model...")
    from llama_cpp import Llama
    sql_llm = Llama(
        model_path=config['models']['sql']['path'],
        n_ctx=config['models']['sql']['n_ctx'],
        n_gpu_layers=config['models']['sql']['n_gpu_layers'],
        verbose=False
    )
    
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "port": os.getenv("DB_PORT", "5432")
    }
    
    indian_states = fetch_all_states(db_config)
    indian_districts = fetch_all_districts(db_config)
    
    import gc
    
    # Read all input queries
    all_rows = []
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Total queries in input: {len(all_rows)}")
    processed_count = 0

    for row in all_rows:
        # Refresh completed questions at start of each iteration
        completed_questions = set()
        if os.path.exists(output_csv):
            with open(output_csv, 'r', encoding='utf-8-sig') as f:
                comp_reader = csv.DictReader(f)
                for c_row in comp_reader:
                    q = c_row.get('sql_queries')
                    if q: completed_questions.add(q.strip())

        target_query = (row.get('sql_queries') or row.get('question') or row.get('query') or row.get('prompt', '')).strip()
        if not target_query or target_query in completed_questions:
            continue

        print(f"\n[{processed_count+1}] Processing: {target_query[:50]}...")
        target_gold = row.get('gold_value')
        target_gold_sql = row.get('gold_sql')
        start_time = time.time()
        
        # Location Routing
        detected_state = None
        detected_district = None
        lower_input = target_query.lower()
        for state in indian_states:
            if state in lower_input:
                detected_state = state
                break
        
        if not detected_state:
            for dist in indian_districts:
                if re.search(rf'\b{re.escape(dist)}\b', lower_input):
                    detected_district = dist
                    break
        
        ignored_keywords = [
            "what", "where", "temperature", "rainfall", "min", "max", "average", "lowest", "highest", 
            "location", "district", "state", "india", "weather", "forecast", "in", "of", "the", 
            "distinct", "year", "month", "how", "is", "at", "level", "recorded", "type", "are", 
            "there", "database", "all", "show", "give", "list", "tell", "me", "values", "grouped", "by"
        ]
        words = re.findall(r'\b[a-zA-Z]+\b', target_query)
        relevant_words = [w for w in words if w.lower() not in ignored_keywords]

        if detected_state:
            example_name = detected_state.title()
            pruned = get_pruned_schema("STATE", "")
            prompt = STATE_PROMPT_TEMPLATE.format(pruned_schema=pruned, example_location_name=example_name, user_input=target_query)
            mode = "STATE"
        elif detected_district:
            example_name = detected_district.title()
            pruned = get_pruned_schema("DISTRICT", "")
            prompt = DISTRICT_PROMPT_TEMPLATE.format(pruned_schema=pruned, example_location_name=example_name, user_input=target_query)
            mode = "DISTRICT"
        else:
            pruned = get_pruned_schema("GENERAL", "")
            prompt = GENERAL_SQL_PROMPT_TEMPLATE.format(pruned_schema=pruned, user_input=target_query)
            mode = "GENERAL"
            example_name = "N/A"

        # 1. SQL Generation (Fresh init for every query)
        from llama_cpp import Llama
        print("   Initializing SQL Model...")
        sql_llm = Llama(
            model_path=config['models']['sql']['path'],
            n_ctx=config['models']['sql']['n_ctx'],
            n_gpu_layers=config['models']['sql']['n_gpu_layers'],
            verbose=False
        )
        
        output = sql_llm(prompt, max_tokens=256, stop=["```", "<|im_end|>"], echo=False, temperature=0.1)
        pred_sql_raw = output['choices'][0]['text'].strip()
        pred_sql = fix_generated_sql(pred_sql_raw)
        
        # Clear model memory explicitly
        del sql_llm
        gc.collect()
        print("   Model released and memory cleared.")

        # 2. SQL Execution
        model_answer = "N/A"
        try:
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute(pred_sql)
            rows = cur.fetchall()
            if rows:
                if len(rows) == 1 and len(rows[0]) == 1:
                    model_answer = str(rows[0][0])
                else:
                    model_answer = str(rows[0])
            else:
                model_answer = "Empty Result"
            conn.close()
        except Exception as e:
            model_answer = f"Error: {str(e)}"

        total_time = time.time() - start_time
        
        # 3. Accuracy Comparison
        is_match = False
        if target_gold is not None and model_answer != "N/A":
            g_str = str(target_gold).strip().lower()
            m_str = str(model_answer).strip().lower()
            if g_str == m_str:
                is_match = True
            else:
                try:
                    g_val = float(g_str)
                    m_val = float(m_str)
                    is_match = abs(g_val - m_val) < 0.01
                except (ValueError, TypeError):
                    pass
        
        # Save Result (Append mode)
        file_exists = os.path.exists(output_csv)
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "sql_queries": target_query,
                "gold_value": target_gold if target_gold else "N/A",
                "model_answer": model_answer,
                "match": is_match,
                "gold_sql": target_gold_sql if target_gold_sql else "N/A",
                "model_sql": pred_sql,
                "timing(in secs)": round(total_time, 2),
                "Mode": mode,
                "Detected_Entity": example_name
            })
        
        processed_count += 1
        print(f"   Done. Match: {is_match}. Total Time: {total_time:.2f}s")

    print(f"\n✅ Benchmark Loop Complete. Processed {processed_count} new queries.")

if __name__ == "__main__":
    run_benchmark()
