import os
import json
import re
import csv
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def parse_testing_json(file_path):
    """
    Robustly parses a file that contains multiple JSON lists.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all [...] blocks
    # This regex is a bit simplistic but works if nesting is not deep or if we just want top-level lists
    # A more robust way is to count brackets
    all_cases = []
    
    # Try finding all [ ... ] blocks
    # Using a non-greedy match across newlines
    blocks = re.findall(r'\[.*?\]', content, re.DOTALL)
    
    for block in blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                all_cases.extend(data)
        except json.JSONDecodeError:
            # If a block fails, we might have nested brackets, skipping for now
            # but let's try to be a bit more robust if needed
            continue
            
    return all_cases

def sanitize_string(s):
    """
    Removes newlines, tabs, and extra whitespace to keep CSV cells neat.
    """
    if not s:
        return ""
    # Replace all whitespace sequences (newlines, tabs, multiple spaces) with a single space
    return re.sub(r'\s+', ' ', str(s)).strip()

def run_benchmark():
    input_file = "/Users/pulkitchauhan/Desktop/IDEAS_TIH/intent_clasifier/data/testing.json"
    output_file = "/Users/pulkitchauhan/Desktop/IDEAS_TIH/intent_clasifier/data/benchmark_output_final.csv"
    
    print(f"Reading test cases from {input_file}...", flush=True)
    test_cases = parse_testing_json(input_file)
    print(f"Found {len(test_cases)} test cases.", flush=True)
    
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "pulkitchauhan"),
        "password": os.getenv("DB_PASSWORD", ""),
        "port": os.getenv("DB_PORT", "5432")
    }
    
    results = []
    
    try:
        print(f"Connecting to database {db_config['database']}...", flush=True)
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        for i, case in enumerate(test_cases):
            question = sanitize_string(case.get("question", "N/A"))
            # Handle both "sql" and "query" keys
            sql_raw = case.get("sql") or case.get("query", "")
            sql = sanitize_string(sql_raw)
            
            print(f"[{i+1}/{len(test_cases)}] Running: {question[:50]}...", flush=True)
            
            sql_result = ""
            try:
                if sql_raw:
                    cur.execute(sql_raw)
                    if cur.description:
                        rows = cur.fetchall()
                        # Format rows as string representation
                        sql_result = str(rows)
                        
                        # TRUNCATION: If result is too long, truncate it to keep CSV neat
                        # 1000 characters is usually enough for a preview in a single cell
                        if len(sql_result) > 1000:
                            sql_result = sql_result[:997] + "..."
                    else:
                        conn.commit()
                        sql_result = "Success (No results)"
                else:
                    sql_result = "Error: No SQL provided"
            except Exception as e:
                sql_result = f"Error: {str(e)}"
                conn.rollback() # Rollback on error to continue
            
            results.append({
                "question": question,
                "sql": sql,
                "sql result": sanitize_string(sql_result)
            })
            
        conn.close()
        
    except Exception as e:
        print(f"Fatal DB Error: {e}", flush=True)
    
    print(f"Writing results to {output_file}...", flush=True)
    # Using excel dialect to ensure maximum compatibility with spreadsheet viewers
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "sql", "sql result"], dialect='excel')
        writer.writeheader()
        writer.writerows(results)
    
    print("Done!", flush=True)

if __name__ == "__main__":
    run_benchmark()
