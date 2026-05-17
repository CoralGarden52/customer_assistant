"""Preprocess manual txt files into clean text for GraphRAG."""
import json
import os
import re

input_dir = "/home/wyz/kefu/data/手册"
output_dir = "/home/wyz/kefu/graphrag_project/input"

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".txt"):
        continue
    src = os.path.join(input_dir, fname)
    with open(src, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Parse JSON array: ["text", ["image_ids"]]
    try:
        parsed = json.loads(raw)
        text = parsed[0] if isinstance(parsed, list) else raw
    except json.JSONDecodeError:
        text = raw

    # Clean up: replace literal \n with actual newlines, remove <PIC> markers
    text = text.replace("\\n", "\n")
    text = re.sub(r"<PIC>", "", text)
    # Clean up multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    dst = os.path.join(output_dir, fname)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Processed: {fname} ({len(text)} chars)")
