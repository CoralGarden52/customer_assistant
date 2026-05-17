"""Sort submission.csv by ID in ascending order."""
import csv

INPUT = "submission.csv"


def main():
    rows = []
    with open(INPUT, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["id"]), row["ret"]))

    rows.sort(key=lambda x: x[0])

    with open(INPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        for qid, ret in rows:
            writer.writerow({"id": qid, "ret": ret})

    print(f"Sorted {len(rows)} rows: ID {rows[0][0]}-{rows[-1][0]}")


if __name__ == "__main__":
    main()
