
import sys
import os
import time

import getpass
import psycopg2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from multiclass_intent_classifier import MulticlassIntentClassifier
import re

def fix_generated_sql(sql):
    """
    Robustly fixes common SQL generation errors, specifically missing schema prefixes.
    Ignores keys that are inside single quotes (string literals).
    """
    # List of all tables in the schema
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
    
    # Strategy: Split SQL by single quotes to separate code from string literals
    # We only apply replacements to the EVEN indexed parts (0, 2, 4...) which are outside quotes
    parts = sql.split("'")
    
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        
        # 1. Fix Schema Prefixes
        # 1. Fix Schema Prefixes
        for table in tables:
            # Case A: Table has prefix but NO quotes (e.g. npcyf_schema.DISTRICT -> npcyf_schema."DISTRICT")
            # We match npcyf_schema.TABLE (case insensitive) and replace with quoted version
            pattern_unquoted = re.compile(f'npcyf_schema\.{table}\\b', re.IGNORECASE)
            # Use a lambda or just fixed string since we know the table name
            # We must be careful not to double-quote if it already has quotes ( regex below helps)
            
            # Actually, simpler loop:
            # First, check for prefix WITHOUT quotes
            pattern_unquoted = re.compile(f'npcyf_schema\.{table}\\b(?!\")', re.IGNORECASE) 
            chunk = pattern_unquoted.sub(f'npcyf_schema."{table}"', chunk)

            # Case B: Table has NO prefix (e.g. DISTRICT -> npcyf_schema."DISTRICT")
            # Must not match if it already has prefix (quoted or unquoted)
            pattern_no_prefix = re.compile(f'(?<!npcyf_schema\.)(?<!npcyf_schema\.")\\b{table}\\b', re.IGNORECASE)
            chunk = pattern_no_prefix.sub(f'npcyf_schema."{table}"', chunk)
        
        # Removed risky T2->T1 regex alias fix
        
        parts[i] = chunk
        
    return "'".join(parts)

def get_db_credentials():
    """
    Retrieve database credentials from environment variables or interactive prompt.
    Returns a dictionary with credentials.
    """
    print("\n🔐 Checking for Database Credentials...")
    
    # Priority: Environment Variables
    creds = {
        "host": os.environ.get("DB_HOST"),
        "database": os.environ.get("DB_NAME"),
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
        "port": os.environ.get("DB_PORT")
    }

    # Check if critical credentials are present in env (password might be empty/optional for local)
    if creds["host"] and creds["database"] and creds["user"]:
         print("✅ Credentials found in environment variables.")
         return creds

    print("⚠️  Environment variables missing (or incomplete). Using defaults/prompting.")
    
    # Defaults from User Context
    default_host = "localhost"
    default_db = "postgres"
    default_user = "pulkitchauhan"
    default_port = "5432"

    # Flush any buffered input
    try:
        import termios, sys
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except:
        pass
    
    # Use distinct prompts to avoid buffer confusion
    print(f"\nPress Enter to accept defaults, or type new values.")
    
    h = input(f"Host [{default_host}]: ").strip()
    creds["host"] = h if h else default_host
    
    d = input(f"Database [{default_db}]: ").strip()
    creds["database"] = d if d else default_db
    
    u = input(f"User [{default_user}]: ").strip()
    creds["user"] = u if u else default_user
    
    # Password often empty for local dev
    p = getpass.getpass("Password [None]: ").strip()
    creds["password"] = p if p else None
    
    po = input(f"Port [{default_port}]: ").strip()
    creds["port"] = po if po else default_port
    
    return creds

def execute_sql_query(sql_query, db_config):
    """
    Execute a SQL query against the PostgreSQL database.
    """
    conn = None
    try:
        # Clean SQL (remove markdown code blocks if present)
        clean_sql = sql_query.replace("```sql", "").replace("```", "").strip()
        
        print(f"\n🔌 Connecting to database '{db_config['database']}' at {db_config['host']}...")
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        print(f"🚀 Executing Query...")
        cur.execute(clean_sql)
        
        if cur.description:
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            
            print(f"\n📊 Query Results ({len(rows)} rows):")
            print("-" * 60)
            print(" | ".join(columns))
            print("-" * 60)
            for row in rows[:20]: # Limit to 20 rows for display
                print(" | ".join(map(str, row)))
            if len(rows) > 20:
                print(f"... and {len(rows) - 20} more rows.")
            print("-" * 60)
            return columns, rows
        else:
            conn.commit()
            print("✅ Query executed successfully (no results to return).")
            return None, None
            
    except Exception as e:
        print(f"\n❌ Database Error: {e}")
        return None, None
    finally:
        if conn:
            conn.close()

def interactive_mode():
    # Initialize LLMs
    general_llm = None
    sql_llm = None
    db_config = None # Store credentials once retrieved
    
    try:
        from llama_cpp import Llama
        
        # Paths - Update these or place models here
        GENERAL_MODEL_PATH = "models/general_model.gguf"
        # Using Qwen 1.5B GGUF for SQL generation (fits in memory)
        SQL_MODEL_PATH = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
        
        if os.path.exists(GENERAL_MODEL_PATH):
            print(f"Loading General Model from {GENERAL_MODEL_PATH}...")
            # Optimized for Apple Silicon (Metal)
            general_llm = Llama(
                model_path=GENERAL_MODEL_PATH, 
                n_gpu_layers=-1, # Offload all layers to GPU
                n_ctx=2048,      # Reasonable context for chat
                n_batch=256,     # Reduced for memory stability
                flash_attn=True, # Enable Flash Attention
                verbose=False
            )
        else:
            print(f"Warning: General model not found at {GENERAL_MODEL_PATH}. General queries will only show intent.")
            
        if os.path.exists(SQL_MODEL_PATH):
            print(f"Loading SQL Model from {SQL_MODEL_PATH}...")
            # Optimized for Apple Silicon (Metal) & SQL Schema
            sql_llm = Llama(
                model_path=SQL_MODEL_PATH, 
                n_gpu_layers=-1, # Offload all layers to GPU
                n_ctx=8192,      # Large context for Schema (~4k tokens) + Query
                n_batch=256,     # Reduced for memory stability
                flash_attn=True, # Enable Flash Attention
                verbose=False
            )
        else:
            print(f"Warning: SQL model not found at {SQL_MODEL_PATH}. SQL queries will only show intent.")
            
    except ImportError:
        print("Warning: llama-cpp-python not installed. LLM generation disabled.")
    except Exception as e:
        print(f"Error initializing LLMs: {e}")

    # Load Intent Classifier
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

    # Load Database Schema
    schema_path = os.path.join(os.path.dirname(__file__), '../resourcesfortrainingtheintentclassifier/Database Schema.sql')
    schema_content = ""
    try:
        with open(schema_path, 'r') as f:
            schema_content = f.read()
        print(f"Schema loaded from {schema_path}")
    except Exception as e:
        print(f"Warning: Could not load schema from {schema_path}: {e}")

    # Conversation history for General LLM
    history = []

    while True:
        try:
            user_input = input("\n📝 Enter Query: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting... Goodbye!")
                break
            
            if not user_input:
                continue
            
            start_time = time.time()
            
            # Classification Timing
            clf_start = time.time()
            result = clf.predict(user_input)
            clf_end = time.time()
            clf_time_ms = (clf_end - clf_start) * 1000
            
            intent = result['intent']
            confidence = result['confidence']
            
            print(f"🤖 Prediction: {intent.upper()}")
            print(f"📊 Confidence: {confidence:.2%}")
            print(f"⏱️  Classification Time: {clf_time_ms:.2f} ms")
            
            # Update history with user input
            history.append(f"User: {user_input}")
            
            # Routing Logic
            if intent == "general":
                if general_llm:
                    print(f"🧠 Generating response from General LLM...")
                    output = general_llm(
                        f"User: {user_input}\nAssistant:", 
                        max_tokens=256, 
                        stop=["User:", "\n"], 
                        echo=False
                    )
                    response_text = output['choices'][0]['text'].strip()
                    print(f"💬 Response: {response_text}")
                    history.append(f"Assistant: {response_text}")
                else:
                    print("⚠️  General LLM not loaded.")
            
            elif intent == "sql":
                if sql_llm:
                    print(f"🧠 Generating SQL from Text2SQL LLM...")
                    # Enhanced prompt with schema and history
                    history_context = "\n".join(history[-4:]) # Last few turns for context
                    
                    # Construct Prompt with Schema
                    prompt = f"""<|im_start|>system
You are an expert SQL assistant. Your goal is to generate accurate SQL queries based EXACTLY on the provided schema.

CRITICAL SCHEMA RULES (READ CAREFULLY):

1. **DISTRICT QUERY PATTERN (The Most Important Rule)**:
   - `LOCATION` tables DO NOT have `district_name`. You MUST join `DISTRICT` table.
   - ✅ RIGHT: 
     ```sql
     FROM npcyf_schema."LOCATION_TEMPERATURE" AS lt
     JOIN npcyf_schema."DISTRICT" AS d ON lt.location_type = 'DISTRICT' AND lt.location_id = d.district_id
     WHERE d.district_name ILIKE '%Kolkata%'
     ```
   - ❌ WRONG: `WHERE district_name = 'Kolkata'` (Column does not exist in Location tables!)

2. **Schema Prefix**: ALWAYS use `npcyf_schema` prefix for ALL tables.

3. **Polymorphic Location**: Weather tables use `location_type` and `location_id`.



4. **CORRECT JOIN PATTERN (DISTRICT)**:
   ✅ RIGHT (Use semantic aliases `lt`, `d`, `s`):
   ```sql
   FROM npcyf_schema."LOCATION_TEMPERATURE" AS lt
   JOIN npcyf_schema."DISTRICT" AS d ON lt.location_type = 'DISTRICT' AND lt.location_id = d.district_id
   JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id
   WHERE d.district_name ILIKE '%Kolkata%'
   ```

   ⚠️ **ALIAS CONSISTENCY (CRITICAL)**:
   - For `LOCATION_TEMPERATURE`, alias MUST be `lt`. Select `lt.temperature_value`.
   - For `LOCATION_RAINFALL`, alias MUST be `lr`. Select `lr.rainfall_value`.
   - ❌ WRONG: `FROM LOCATION_RAINFALL AS lr SELECT lt.rainfall_value` (lt is undefined!).

   ❌ WRONG (DO NOT DO THIS):
   - **DO NOT Use T1, T2 aliases**: Confusion causes errors. Use `lt`, `d`, `s`.
   - **DO NOT Select from Dimension**: `SELECT d.temperature_value` is WRONG. Use `lt.temperature_value`.

5. **String Matching**: 
   - If filtering by **State Name**: Use `s.state_name ILIKE '%Name%'`.
   - If filtering by **District Name**: Use `d.district_name ILIKE '%Name%'` (NEVER use `state_name` for a district!).

6. **Date Handling**: 
   - Use `*_recorded_year` columns for specific years (e.g. `rainfall_recorded_year = 2010`).
   - Use `EXTRACT(MONTH FROM ...)` for month filtering.

7. **Aggregation & Grouping**:
   - If using `GROUP BY`, **ALL** selected columns must be aggregated (SUM, AVG) or in the group clause.
   - ❌ WRONG: `SELECT state_name, rainfall_value ... GROUP BY state_name`
   - ✅ RIGHT: `SELECT state_name, SUM(rainfall_value) ... GROUP BY state_name`

8. **IMPOSSIBLE DATA (DO NOT HALLUCINATE)**:
   - **NO Yield Data**: There is NO crop yield column. Output "I cannot answer this (No Yield Data)."
   - **NO Cross-Weather Joins**: Do NOT join `LOCATION_RAINFALL` and `LOCATION_TEMPERATURE`. They are separate.



Schema Definition:
{schema_content}

History:
{history_context}
<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
```sql
"""
                    output = sql_llm(
                        prompt, 
                        max_tokens=256, 
                        stop=["```", "<|im_end|>"], 
                        echo=False,
                        temperature=0.1 # Low temp for precision
                    )
                    sql_query = output['choices'][0]['text'].strip()
                    print(f"💾 SQL: {sql_query}")
                    history.append(f"Assistant: Provided SQL: {sql_query}")

                    # Execute SQL
                    try:
                        if db_config is None:
                            db_config = get_db_credentials()
                        
                        # Fix SQL before execution
                        fixed_sql = fix_generated_sql(sql_query)
                        if fixed_sql != sql_query:
                            print(f"🔧 Fixed SQL: {fixed_sql}")
                        
                        columns, rows = execute_sql_query(fixed_sql, db_config)
                        
                        # --- RETRY LOGIC FOR EMPTY RESULTS ---
                        # Check if rows is empty OR if the single result is None (valid SQL but no data match)
                        param_is_empty = False
                        if rows is not None:
                            if len(rows) == 0:
                                param_is_empty = True
                            elif len(rows) == 1 and (rows[0][0] is None):
                                param_is_empty = True

                            # --- FRESH PROMPT RETRY STRATEGY ---
                            # Instead of appending to history (which traps the model), we use a fresh "Repair" prompt.
                            print("🔄 Generating Retry Prompt...")
                            
                            # Pre-calculate the error message variable to avoid f-string collision with regex/json braces
                            likely_entity = "State" if "state_name" in sql_query else "District"
                            
                            repair_system_prompt = f"""<|im_start|>system
You are a SQL Repair Agent.
The user's previous query returned 0 results because it likely checked the WRONG Location Table (State vs District).

YOUR JOB: Swap the table and column used for filtering location name.

RULES:
1. If original used `state_name` -> CHANGE TO `district_name` and ensure `DISTRICT` table is joined.
2. If original used `district_name` -> CHANGE TO `state_name` and ensure `STATE` table is joined.
3. Keep the rest of the query (aggregations, year filters) the same.
4. Output ONLY the SQL.

Schema:
- `DISTRICT` table has `district_name`.
- `STATE` table has `state_name`.
<|im_end|>
<|im_start|>user
FAILED SQL:
```sql
{{sql_query}}
```

ERROR: No results found. The location is likely not a {likely_entity}.
TASK: Rewrite the query to check the OTHER table.
<|im_end|>
<|im_start|>assistant
```sql
"""
                            # Use format with the properly escaped string
                            formatted_retry_prompt = repair_system_prompt.format(sql_query=sql_query)
                            
                            # Debug print (optional)
                            # print(f"DEBUG RETRY PROMPT:\n{formatted_retry_prompt}")

                            output_retry = sql_llm(
                                formatted_retry_prompt, 
                                max_tokens=256, 
                                stop=["```", "<|im_end|>"], 
                                echo=False,
                                temperature=0.1
                            )
                            retry_sql = output_retry['choices'][0]['text'].strip()
                            print(f"🔄 Retry SQL: {retry_sql}")
                            
                            # Fix and Execute Retry
                            fixed_retry_sql = fix_generated_sql(retry_sql)
                            if fixed_retry_sql != retry_sql:
                                print(f"🔧 Fixed Retry SQL: {fixed_retry_sql}")
                                
                            execute_sql_query(fixed_retry_sql, db_config)
                            
                    except Exception as e:
                       print(f"⚠️ Failed to execute SQL: {e}")

                else:
                     print("⚠️  SQL LLM not loaded.")
            
            elif intent == "platform":
                print("ℹ️  Platform query detected. Refer to documentation.")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"⏱️  Total Time Taken: {elapsed_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\nExiting... Goodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    interactive_mode()
