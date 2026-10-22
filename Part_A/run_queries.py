import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")

# read the file and split it into separate queries at each ";"
text = open("sql/queries.sql", encoding="utf-8").read()
queries = [q.strip() for q in text.split(";") if q.strip()]

for i, q in enumerate(queries, start=1):
    result = pd.read_sql_query(q, conn)
    print(f"\n===== Q{i}: {len(result)} rows =====")
    print(result.head(5))
    print(result.tail(1))
    if "rnk" in result.columns:
        print("rank differs from row_number:", (result["rnk"] != result["row_num"]).sum())

conn.close()