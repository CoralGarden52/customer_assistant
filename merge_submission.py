"""Merge Chinese and English submissions into final submission.csv."""
import csv
from pathlib import Path

ZH_CSV = Path("/home/wyz/kefu/graphrag_zh/submission_zh.csv")
EN_CSV = Path("/home/wyz/kefu/graphrag_en/submission_en.csv")
OUT_CSV = Path("/home/wyz/kefu/data/submission.csv")

results = []

# Chinese results (id 1-240)
if ZH_CSV.exists():
    with open(ZH_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.append({"id": int(row["id"]), "ret": row["ret"]})
    print(f"Chinese: {len([r for r in results])} rows from {ZH_CSV}")
else:
    print(f"WARNING: {ZH_CSV} not found")

# English results (id 241+)
en_count = 0
if EN_CSV.exists():
    with open(EN_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.append({"id": int(row["id"]), "ret": row["ret"]})
            en_count += 1
    print(f"English: {en_count} rows from {EN_CSV}")
else:
    print(f"WARNING: {EN_CSV} not found")

# Sort by id
results.sort(key=lambda x: x["id"])

# Write
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "ret"])
    writer.writeheader()
    writer.writerows(results)

print(f"Merged: {len(results)} rows -> {OUT_CSV}")
