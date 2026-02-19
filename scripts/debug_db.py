import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_credentials():
    creds = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "database": os.environ.get("DB_NAME", "postgres"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASSWORD", None),
        "port": os.environ.get("DB_PORT", "5432")
    }
    return creds

def run_debug():
    config = get_db_credentials()
    try:
        conn = psycopg2.connect(**config)
        cur = conn.cursor()
        
        print("--- Checking 'Bihar' in STATE table ---")
        cur.execute("SELECT state_id, state_name FROM npcyf_schema.\"STATE\" WHERE state_name ILIKE '%Bihar%'")
        rows = cur.fetchall()
        for r in rows:
            print(f"ID: {r[0]}, Name: '{r[1]}'")
            
        if not rows:
            print("❌ No state found matching %Bihar%")
            return

        state_id = rows[0][0]
        
        print("\n--- Checking Year Column Population ---")
        # Check if rainfall_recorded_year is NULL
        query = f"""
        SELECT COUNT(*) 
        FROM npcyf_schema."LOCATION_RAINFALL" lr
        JOIN npcyf_schema."DISTRICT" d ON lr.location_type = 'DISTRICT' AND lr.location_id = d.district_id
        WHERE d.state_id = {state_id} AND lr.rainfall_recorded_year IS NULL
        """
        cur.execute(query)
        null_count = cur.fetchone()[0]
        print(f"Rows with NULL rainfall_recorded_year for Bihar: {null_count}")
        
        print("\n--- Checking Data for 2010 ---")
        # Check count using Year Column
        query_col = f"""
        SELECT COUNT(*) 
        FROM npcyf_schema."LOCATION_RAINFALL" lr
        JOIN npcyf_schema."DISTRICT" d ON lr.location_type = 'DISTRICT' AND lr.location_id = d.district_id
        WHERE d.state_id = {state_id} AND lr.rainfall_recorded_year = 2010
        """
        cur.execute(query_col)
        count_col = cur.fetchone()[0]
        print(f"Count using `rainfall_recorded_year = 2010`: {count_col}")
        
        # Check count using Extract
        query_ext = f"""
        SELECT COUNT(*) 
        FROM npcyf_schema."LOCATION_RAINFALL" lr
        JOIN npcyf_schema."DISTRICT" d ON lr.location_type = 'DISTRICT' AND lr.location_id = d.district_id
        WHERE d.state_id = {state_id} AND EXTRACT(YEAR FROM lr.rainfall_recorded_date) = 2010
        """
        cur.execute(query_ext)
        count_ext = cur.fetchone()[0]
        print(f"Count using `EXTRACT(YEAR...) = 2010`: {count_ext}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    run_debug()
