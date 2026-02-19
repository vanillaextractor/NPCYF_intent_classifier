
import sys
import os
import time

import json
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

import getpass
import psycopg2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from multiclass_intent_classifier import MulticlassIntentClassifier
import re


class SchemaValidator:
    def __init__(self):
        self.schema = {
            "CROP": ["crop_id", "crop_name"],
            "SEASON": ["season_id", "season_name", "season_start", "season_end"],
            "STATE": ["state_id", "state_name", "state_latitude", "state_longitude", "state_area"],
            "DISTRICT": ["district_id", "district_name", "state_id", "district_latitude", "district_longitude", "district_area"],
            "LOCATION_RAINFALL": ["location_rainfall_id", "location_type", "location_id", "rainfall_value", "rainfall_recorded_date", "rainfall_recorded_year", "rainfall_data_collection_id"],
            "LOCATION_TEMPERATURE": ["location_temperature_id", "location_type", "location_id", "temperature_value", "temperature_recorded_date", "temperature_recorded_year", "temperature_data_collection_id"],
            "LOCATION_RESERVOIR_LEVEL": ["location_reservoir_level_id", "location_type", "location_id", "frl", "level", "current_live_storage", "reservoir_level_recorded_date", "reservoir_level_recorded_year", "reservoir_level_data_collection_id"],
            "RAINFALL_DATA_COLLECTION": ["rainfall_data_collection_id", "rainfall_data_collection_rainfall_unit", "rainfall_data_collection_rainfall_recorded_interval_unit"],
            "TEMPERATURE_DATA_COLLECTION": ["temperature_data_collection_id", "temperature_data_collection_temperature_type", "temperature_data_collection_temperature_unit", "temperature_data_collection_temperature_recorded_interval_unit"],
            "RESERVOIR_LEVEL_DATA_COLLECTION": ["reservoir_level_data_collection_id"],
            "FEATURE_MASTER": ["feature_id", "feature_title", "feature_description", "feature_data_category", "feature_temporal_interval", "feature_location_level", "feature_is_active"],
        }
        self.tables = list(self.schema.keys())

    def extract_table_aliases(self, sql):
        # Extract table aliases: FROM table AS alias OR JOIN table [AS] alias
        # This is a basic regex and might need refinement for complex cases, but covers standard SQL generation
        aliases = {}
        
        # Normalize SQL
        sql_lower = sql.lower().replace('\n', ' ')
        
        for table in self.tables:
            # Pattern 1: FROM/JOIN npcyf_schema."TABLE" [AS] alias
            # We look for the table name, optionally quoted, with optional schema prefix
            table_pattern = re.compile(rf'(?:from|join)\s+(?:npcyf_schema\.)?"?{table}"?\s+(?:as\s+)?(\w+)', re.IGNORECASE)
            matches = table_pattern.findall(sql)
            for alias in matches:
                # exclude keywords like 'WHERE', 'ON', 'JOIN', 'AND' if they are matched by mistake (unlikely with \w+ but possible if query is malformed)
                if alias.lower() not in ['where', 'on', 'join', 'and', 'order', 'group', 'limit', 'select']:
                    aliases[alias] = table
            
            # Pattern 2: FROM/JOIN npcyf_schema."TABLE" (No alias - implicit alias is table name?)
            # Actually, if no alias, we can't easily validate T2.col unless T2 is the table name. 
            # But the prompt enforces aliases usually.
            
        return aliases

    def validate(self, sql):
        errors = []
        aliases = self.extract_table_aliases(sql)
        
        # Check 1: Column Existence
        # Look for pattern: alias.column
        # We scan the SQL for "word.word"
        column_refs = re.findall(r'(\w+)\.(\w+)', sql)
        
        for alias, column in column_refs:
            # Skip if prefix is not a known alias (could be schema prefix npcyf_schema.TABLE)
            if alias.lower() == 'npcyf_schema':
                continue
            
            if alias in aliases:
                table = aliases[alias]
                if column not in self.schema[table]:
                    # Special check for case-insensitivity or common hallucinations
                    errors.append(f"❌ Column '{column}' does not exist in table '{table}' (aliased as '{alias}'). Available columns: {', '.join(self.schema[table])}")
            else:
                 pass
                 # If alias not found, it might be a valid alias we missed or a CTE. 
                 # We'll skip strict validation to avoid false positives on complex queries, 
                 # but for the focus tasks (simple joins), this should catch T2.temperature_value on STATE table.

        # Check 2: Date handling
        # User query had: "LIKE '2000-06-%'" which acts on string. If column is DATE, this might fail in some DBs or logic.
        # But Postgres DATE supports LIKE if cast to char, OR we should use EXTRACT.
        # The specific error seen was: "operator does not exist: date ~~ unknown" (~~ is LIKE)
        # So we must forbid LIKE on DATE columns generally without casting.
        
        if " like " in sql.lower() or " ilike " in sql.lower():
            # Check if it's applied to a date column
            for alias, table in aliases.items():
                for date_col in [c for c in self.schema[table] if 'date' in c]:
                    pattern = re.compile(rf'{alias}\.{date_col}\s*i?like', re.IGNORECASE)
                    if pattern.search(sql):
                        errors.append(f"❌ Invalid DATE operation: Do not use LIKE/ILIKE with date column '{alias}.{date_col}'. Use `EXTRACT(MONTH FROM ...)` or `to_char(...)`.")

        return errors


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
        for table in tables:
            # Case A: Table has prefix but NO quotes (e.g. npcyf_schema.DISTRICT -> npcyf_schema."DISTRICT")
            pattern_unquoted = re.compile(rf'npcyf_schema\.{table}\b(?!\")', re.IGNORECASE) 
            chunk = pattern_unquoted.sub(f'npcyf_schema."{table}"', chunk)

            # Case B: Table has NO prefix (e.g. DISTRICT -> npcyf_schema."DISTRICT")
            pattern_no_prefix = re.compile(rf'(?<!npcyf_schema\.)(?<!npcyf_schema\.")\b{table}\b', re.IGNORECASE)
            chunk = pattern_no_prefix.sub(f'npcyf_schema."{table}"', chunk)
        
        parts[i] = chunk
        
    return "'".join(parts)

def get_db_credentials():
    print("\n🔐 Checking for Database Credentials...")
    
    # Prioritize Environment Variables (loaded from .env)
    creds = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "database": os.environ.get("DB_NAME", "postgres"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASSWORD", None),
        "port": os.environ.get("DB_PORT", "5432")
    }

    # If password is set in env, we can likely skip the prompt (or if user doesn't want to use password)
    # Checks if critical params are present.
    if creds["user"] and creds["database"]:
         print(f"   Detected settings from .env: {creds['user']}@{creds['host']}:{creds['port']}/{creds['database']}")
         return creds
    
    # Use distinct prompts to avoid buffer confusion
    print(f"\nPress Enter to accept defaults, or type new values.")
    
    h = input(f"Host [{creds['host']}]: ").strip()
    creds["host"] = h if h else creds['host']
    
    d = input(f"Database [{creds['database']}]: ").strip()
    creds["database"] = d if d else creds['database']
    
    u = input(f"User [{creds['user']}]: ").strip()
    creds["user"] = u if u else creds['user']
    
    # Password often empty for local dev
    p = getpass.getpass("Password [None]: ").strip()
    creds["password"] = p if p else None
    
    po = input(f"Port [{creds['port']}]: ").strip()
    creds["port"] = po if po else creds['port']
    
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

def fetch_all_states(db_config):
    """
    Fetch all state names from the database.
    """
    conn = None
    states = []
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute('SELECT "state_name" FROM npcyf_schema."STATE"')
        rows = cur.fetchall()
        states = [row[0].lower() for row in rows]
    except Exception as e:
        print(f"Warning: Could not fetch states from database: {e}")
    finally:
        if conn:
            conn.close()
    return states

def fetch_all_districts(db_config):
    """
    Fetch all district names from the database.
    """
    conn = None
    districts = []
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute('SELECT "district_name" FROM npcyf_schema."DISTRICT"')
        rows = cur.fetchall()
        districts = [row[0].lower() for row in rows]
    except Exception as e:
        print(f"Warning: Could not fetch districts from database: {e}")
    finally:
        if conn:
            conn.close()
    return districts

def get_pruned_schema(query_type, full_schema):
    """
    Returns a pruned version of the schema as a string.
    - If query_type is 'DISTRICT', the 'STATE' table and state_id cross-refs are hidden.
    - If query_type is 'STATE', the 'DISTRICT' table is hidden.
    - If query_type is 'GENERAL', returns the full schema.
    """
    pruned = full_schema.copy()
    if query_type == "DISTRICT":
        # Hide STATE table completely
        if "STATE" in pruned: del pruned["STATE"]
        # Hide state_id from DISTRICT (too tempting for LLM to join)
        if "DISTRICT" in pruned:
            pruned["DISTRICT"] = [c for c in pruned["DISTRICT"] if c != "state_id"]
    elif query_type == "STATE":
        # For STATE queries, we NEED DISTRICT table to join fact data (only at district level) to STATE
        pass 
    elif query_type == "GENERAL":
        # No pruning for general queries
        pass
    
    # Format as string for prompt
    schema_str = ""
    for table, cols in pruned.items():
        schema_str += f"- {table}: {', '.join(cols)}\n"
    return schema_str

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
1. **LOCATION_TYPE**: ALWAYS use `location_type = 'DISTRICT'` in the JOIN condition.
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
   - To get STATE name: Join Fact Table -> DISTRICT -> STATE.
   - **Correct Join Chain**: `JOIN npcyf_schema."DISTRICT" AS d ON fact.location_type = 'DISTRICT' AND fact.location_id = d.district_id JOIN npcyf_schema."STATE" AS s ON d.state_id = s.state_id`
4. **AGGREGATION RULES**:
   - For **Rainfall**: Use `SUM(lr.rainfall_value)`.
   - For **Reservoir Level**:
     - "Highest level" -> Use `MAX(lr.level)`.
     - "Total storage" -> Use `SUM(lr.current_live_storage)`.
     - NEVER use `SUM(frl)` for level queries.
5. **NO HALLUCINATION**: There is NO table named `LOCATION`. NEVER use a table named `LOCATION`. Use `STATE` or `DISTRICT` instead.
6. **SCHEMA**: Strictly follow the table names and column names provided.
7. **DATE**: Use `EXTRACT(YEAR/MONTH FROM date_col)`.
8. **MATCHING**: Use `ILIKE` with `%` for flexible matching of names.

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


def interactive_mode():
    # Load Config
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded from config.json")
    except Exception as e:
        print(f"❌ Failed to load config.json: {e}")
        return

    # Initialize LLMs from Config
    general_llm = None
    sql_llm = None
    db_config = None 
    
    # Initialize Schema Validator
    validator = SchemaValidator()

    try:
        from llama_cpp import Llama
        
        # Paths from Config
        GENERAL_MODEL_PATH = config['models']['general']['path']
        SQL_MODEL_PATH = config['models']['sql']['path']
        
        if os.path.exists(GENERAL_MODEL_PATH):
            print(f"Loading General Model from {GENERAL_MODEL_PATH}...")
            general_llm = Llama(
                model_path=GENERAL_MODEL_PATH, 
                n_gpu_layers=config['models']['general']['n_gpu_layers'],
                n_ctx=config['models']['general']['n_ctx'],
                verbose=False
            )
        else:
            print(f"Warning: General model not found at {GENERAL_MODEL_PATH}")
            
        if os.path.exists(SQL_MODEL_PATH):
            print(f"Loading SQL Model from {SQL_MODEL_PATH}...")
            sql_llm = Llama(
                model_path=SQL_MODEL_PATH, 
                n_gpu_layers=config['models']['sql']['n_gpu_layers'],
                n_ctx=config['models']['sql']['n_ctx'],
                verbose=False
            )
        else:
            print(f"Warning: SQL model not found at {SQL_MODEL_PATH}")
            
    except ImportError:
        print("Warning: llama-cpp-python not installed.")
    except Exception as e:
        print(f"Error initializing LLMs: {e}")

    # Database Config and States Initialization
    db_config = get_db_credentials()
    indian_states = fetch_all_states(db_config)
    indian_districts = fetch_all_districts(db_config)
    
    # Conversation history for General LLM
    history = []

    # Load Intent Classifier
    try:
        clf_path = config['models']['intent_classifier']['path']
        clf = MulticlassIntentClassifier(model_path=clf_path)
        print("Intent Classifier loaded successfully!")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return

    # Load Database Schema
    schema_path = os.path.join(os.path.dirname(__file__), '..', config['paths']['schema'])
    schema_content = ""
    try:
        with open(schema_path, 'r') as f:
            schema_content = f.read()
        print(f"Schema loaded from {schema_path}")
    except Exception as e:
        print(f"Warning: Could not load schema: {e}")

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
                    
                    # --- DYNAMIC LOCATION HINTING ---
                    detected_state = None
                    detected_district = None
                    lower_input = user_input.lower()
                    
                    for state in indian_states:
                        if state in lower_input:
                            detected_state = state
                            break
                    
                    if not detected_state:
                        for dist in indian_districts:
                            # Use word boundaries to avoid catching substrings like 'in' in 'rainfall'
                            if re.search(rf'\b{re.escape(dist)}\b', lower_input):
                                detected_district = dist
                                break
                    
                    # Normalize entity for examples
                    ignored_keywords = [
                        "what", "where", "temperature", "rainfall", "min", "max", "average", "lowest", "highest", 
                        "location", "district", "state", "india", "weather", "forecast", "in", "of", "the", 
                        "distinct", "year", "month", "how", "is", "at", "level", "recorded", "type", "are", 
                        "there", "database", "all", "show", "give", "list", "tell", "me", "values", "grouped", "by"
                    ]
                    words = re.findall(r'\b[a-zA-Z]+\b', user_input)
                    relevant_words = [w for w in words if w.lower() not in ignored_keywords]
                    
                    if detected_state:
                         # STATE PATH
                         example_location_name = detected_state.title()
                         pruned_schema = get_pruned_schema("STATE", validator.schema)
                         prompt = STATE_PROMPT_TEMPLATE.format(
                             pruned_schema=pruned_schema,
                             example_location_name=example_location_name,
                             user_input=user_input
                         )
                    elif detected_district:
                         # DISTRICT PATH
                         example_location_name = detected_district.title()
                         pruned_schema = get_pruned_schema("DISTRICT", validator.schema)
                         prompt = DISTRICT_PROMPT_TEMPLATE.format(
                             pruned_schema=pruned_schema,
                             example_location_name=example_location_name,
                             user_input=user_input
                         )
                    else:
                         # GENERAL PATH (No clear location detected)
                         pruned_schema = get_pruned_schema("GENERAL", validator.schema)
                         prompt = GENERAL_SQL_PROMPT_TEMPLATE.format(
                             pruned_schema=pruned_schema,
                             user_input=user_input
                         )

                    output = sql_llm(
                        prompt, 
                        max_tokens=256, 
                        stop=["```", "<|im_end|>"], 
                        echo=False,
                        temperature=0.1
                    )
                    sql_query = output['choices'][0]['text'].strip()
                    print(f"💾 SQL: {sql_query}")
                    history.append(f"Assistant: Provided SQL and executed query.")

                    # Execute SQL
                    try:
                        if db_config is None:
                            db_config = get_db_credentials()
                        
                        # Fix SQL before execution
                        fixed_sql = fix_generated_sql(sql_query)
                        if fixed_sql != sql_query:
                            print(f"🔧 Fixed SQL: {fixed_sql}")
                        
                        # Validate SQL
                        validation_errors = validator.validate(fixed_sql)
                        if validation_errors:
                            print(f"❌ Validation Errors: {validation_errors}")
                            # Trigger retry with validation errors
                            rows = None # Simulate failure to trigger retry
                        else:
                            columns, rows = execute_sql_query(fixed_sql, db_config)
                        
                        # --- RETRY LOGIC FOR EMPTY RESULTS OR ERRORS ---
                        # Check if rows is empty OR if the single result is None (valid SQL but no data match) OR if validation failed
                        if (rows is not None and (len(rows) == 0 or (len(rows) == 1 and rows[0][0] is None))) or validation_errors:
                            print("🔄 Generating Retry Prompt...")
                            
                            error_context = ""
                            if validation_errors:
                                error_context = "PREVIOUS SQL HAD SCHEMA ERRORS:\n" + "\n".join(validation_errors)
                            else:
                                error_context = "PREVIOUS SQL RETURNED 0 RESULTS. Likely wrong Location Type."

                            # Simplified Repair Prompt (Modular aware)
                            repair_prompt = f"""<|im_start|>system
You are a SQL Repair Agent.
Previous query FAILED or returned 0 results.

REPAIR STRATEGIES:
1. **NO HALLUCINATION**: Check if you used a column that doesn't exist (e.g., `location_type` on `CROP` or `SEASON` table). `location_type` ONLY exists on `LOCATION_RAINFALL`, `LOCATION_TEMPERATURE`, and `LOCATION_RESERVOIR_LEVEL`.
2. **Case Sensitivity**: If results were empty, use `ILIKE` for name matching.
3. **Schema Adherence**: strictly follow the column names in the provided schema.
4. **DATE**: NEVER use `LIKE` on DATE columns. Use `EXTRACT`.
<|im_end|>
<|im_start|>user
FAILED SQL:
```sql
{sql_query}
```
Error/Status: {error_context}
Fix it.
<|im_end|>
<|im_start|>assistant
```sql
"""
                            output_retry = sql_llm(
                                repair_prompt, 
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

                            # Validate again
                            retry_errors = validator.validate(fixed_retry_sql)
                            if retry_errors:
                                print(f"❌ Retry Validation Failed: {retry_errors}")
                            else:
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
