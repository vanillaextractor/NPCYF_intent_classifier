# ==========================================================
# 🔥 PROJECT ROOT FIX (NO IMPORT ERRORS)
# ==========================================================

import sys
import os

PROJECT_ROOT = "/Users/pulkitchauhan/Desktop/IDEAS_TIH/intent_clasifier"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ==========================================================
# IMPORTS
# ==========================================================

import json
import re
import psycopg2
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()

from multiclass_intent_classifier import MulticlassIntentClassifier


# ==========================================================
# CONFIG LOAD
# ==========================================================

CONFIG_PATH = "/Users/pulkitchauhan/Desktop/IDEAS_TIH/intent_clasifier/config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

BASE_DIR = os.path.dirname(CONFIG_PATH)

SCHEMA = "npcyf_schema"

# Only tables that need schema prefix
TABLES = [
    "STATE","DISTRICT","CROP","SEASON",
    "LOCATION_TEMPERATURE",
    "LOCATION_RAINFALL",
    "LOCATION_RESERVOIR_LEVEL"
]


# ==========================================================
# 🔥 LLM FULL SQL GENERATION
# ==========================================================

def generate_full_sql(llm, schema_text, user_query):

    prompt = f"""
You are an expert PostgreSQL SQL generator.

RULES:
- Use only tables from schema
- Follow column names exactly
- Generate FULL SQL
- Do NOT invent columns

SCHEMA:
{schema_text}

USER QUERY:
{user_query}
"""

    out = llm(
        prompt,
        max_tokens=300,
        temperature=0.0,
        stop=["```","\n\n"]
    )

    sql = out['choices'][0]['text'].strip()
    sql = sql.replace("```sql","").replace("```","")

    return sql


# ==========================================================
# 🔥 AUTO SCHEMA PREFIXER (FIX relation does not exist)
# ==========================================================

def apply_schema_prefix(sql):

    for t in TABLES:
        sql = re.sub(
            rf'\b{t}\b',
            f'{SCHEMA}."{t}"',
            sql,
            flags=re.IGNORECASE
        )

    return sql


# ==========================================================
# 🔥 AUTO JOIN REWRITER (POLYMORPHIC SAFE FIX)
# ==========================================================

def rewrite_polymorphic_joins(sql):

    lower_sql = sql.lower()

    # TEMPERATURE
    if "location_temperature" in lower_sql and "join" not in lower_sql:

        sql += f"""
 JOIN {SCHEMA}."STATE" s
   ON lt.location_type='STATE'
  AND lt.location_id=s.state_id
"""

    # RAINFALL
    if "location_rainfall" in lower_sql and "join" not in lower_sql:

        sql += f"""
 JOIN {SCHEMA}."DISTRICT" d
   ON lr.location_type='DISTRICT'
  AND lr.location_id=d.district_id
 JOIN {SCHEMA}."STATE" s
   ON d.state_id=s.state_id
"""

    # RESERVOIR
    if "location_reservoir_level" in lower_sql and "join" not in lower_sql:

        sql += f"""
 JOIN {SCHEMA}."DISTRICT" d
   ON lrl.location_type='DISTRICT'
  AND lrl.location_id=d.district_id
 JOIN {SCHEMA}."STATE" s
   ON d.state_id=s.state_id
"""

    return sql


# ==========================================================
# DB EXECUTION
# ==========================================================

def execute(sql, db):

    try:
        print("\n🚀 Executing Query...")
        conn = psycopg2.connect(**db)
        cur = conn.cursor()

        cur.execute(sql)

        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

        print("\n📊 Results")
        print(" | ".join(cols))
        print("-"*40)

        for r in rows[:20]:
            print(" | ".join(map(str,r)))

        conn.close()

    except Exception as e:
        print("❌ Database Error:", e)


# ==========================================================
# MAIN LOOP
# ==========================================================

def interactive_mode():

    sql_llm = Llama(
        model_path=os.path.join(BASE_DIR, config['models']['sql']['path']),
        n_gpu_layers=config['models']['sql']['n_gpu_layers'],
        n_ctx=config['models']['sql']['n_ctx'],
        verbose=False
    )

    clf = MulticlassIntentClassifier(
        model_path=os.path.join(BASE_DIR, config['models']['intent_classifier']['path'])
    )

    schema_path = os.path.join(BASE_DIR, config['paths']['schema'])
    schema_content = open(schema_path).read()

    db_config = {
        "host":"localhost",
        "database":"postgres",
        "user":"pulkitchauhan",
        "password":None,
        "port":"5432"
    }

    while True:

        user_input = input("\n📝 Enter Query: ").strip()

        if user_input.lower() in ["exit","quit"]:
            break

        result = clf.predict(user_input)
        print("🤖 Prediction:", result['intent'])

        if result['intent'] != "sql":
            continue

        # --------------------------------------------------
        # Generate SQL from LLM
        # --------------------------------------------------

        final_sql = generate_full_sql(
            sql_llm,
            schema_content,
            user_input
        )

        # --------------------------------------------------
        # Fix table schema names automatically
        # --------------------------------------------------

        final_sql = apply_schema_prefix(final_sql)

        # --------------------------------------------------
        # Fix polymorphic joins automatically
        # --------------------------------------------------

        final_sql = rewrite_polymorphic_joins(final_sql)

        print("\n✅ FINAL SQL:\n", final_sql)

        execute(final_sql, db_config)


if __name__ == "__main__":
    interactive_mode()
