import csv
import os

base_dir = "/Users/pulkitchauhan/Desktop/IDEAS_TIH/intent_clasifier/benchmark"
input_csv = os.path.join(base_dir, "input.csv")
output_csv = os.path.join(base_dir, "output.csv")
temp_output = os.path.join(base_dir, "output_temp.csv")

# 1. Map queries to gold_sql from input.csv
query_to_gold_sql = {}
with open(input_csv, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row.get('sql_queries', '').strip()
        if q:
            query_to_gold_sql[q] = row.get('gold_sql', 'N/A')

# 2. Read existing output.csv and add gold_sql
fieldnames = ["sql_queries", "gold_value", "gold_sql", "model_sql", "model_answers", "Match", "timing(in secs)", "Mode", "Detected_Entity"]
new_rows = []

if os.path.exists(output_csv):
    with open(output_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get('sql_queries', '').strip()
            row['gold_sql'] = query_to_gold_sql.get(q, 'N/A')
            new_rows.append(row)

# 3. Rewrite output.csv
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(new_rows)

print(f"Upgrade complete. {len(new_rows)} rows updated in {output_csv}")
