"""Build a stable text_unit_id -> images map from current GraphRAG outputs."""
import json
import re
from pathlib import Path

import pandas as pd

from preprocess_multimodal import find_pic_positions, process_manual

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
SOURCE_DIR = Path("/home/wyz/kefu/data/手册")
MAP_FILE = PROJECT_ROOT / "text_unit_image_map.json"


def build_pic_context(text: str, pic_pos: int, window: int = 150) -> str:
    """Extract local text context around one <PIC> marker."""
    ctx_start = max(0, pic_pos - window)
    ctx_end = min(len(text), pic_pos + window)
    return text[ctx_start:ctx_end].strip()


def find_chunk_span(full_text: str, chunk_text: str, start_hint: int) -> tuple[int, int]:
    """Locate a text_unit inside its source document with overlap-aware search."""
    if not chunk_text:
        return -1, -1

    direct_start = full_text.find(chunk_text, max(0, start_hint))
    if direct_start != -1:
        return direct_start, direct_start + len(chunk_text)

    retry_start = full_text.find(chunk_text, max(0, start_hint - 800))
    if retry_start != -1:
        return retry_start, retry_start + len(chunk_text)

    normalized_full = re.sub(r"\s+", " ", full_text)
    normalized_chunk = re.sub(r"\s+", " ", chunk_text).strip()
    prefix = normalized_chunk[:120]
    if prefix:
        prefix_pos = normalized_full.find(prefix, max(0, start_hint - 800))
        if prefix_pos != -1:
            suffix = normalized_chunk[-120:]
            suffix_pos = normalized_full.find(suffix, prefix_pos)
            if suffix_pos != -1:
                return prefix_pos, suffix_pos + len(suffix)

    return -1, -1


def main():
    documents = pd.read_parquet(OUTPUT_DIR / "documents.parquet")
    text_units = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")
    text_unit_by_id = {str(row["id"]): row for _, row in text_units.iterrows()}

    by_text_unit_id = {}
    by_short_id = {}
    unresolved = []

    for _, doc in documents.iterrows():
        title = str(doc["title"])
        manual_name = title[:-4] if title.endswith(".txt") else title
        source_path = SOURCE_DIR / title
        if not source_path.exists():
            unresolved.append({"document": title, "reason": "source_not_found"})
            continue

        raw_document_text = str(doc.get("text") or "")
        source_text, images = process_manual(str(source_path))
        raw_pic_positions = find_pic_positions(raw_document_text, images)
        processed_pic_positions = find_pic_positions(source_text, images)
        raw_search_start = 0
        processed_search_start = 0

        text_unit_ids = doc["text_unit_ids"]
        if text_unit_ids is None:
            text_unit_ids = []
        elif hasattr(text_unit_ids, "tolist"):
            text_unit_ids = text_unit_ids.tolist()
        else:
            text_unit_ids = list(text_unit_ids)

        for text_unit_id in text_unit_ids:
            text_unit_id = str(text_unit_id)
            row = text_unit_by_id.get(text_unit_id)
            if row is None:
                unresolved.append({"document": title, "text_unit_id": text_unit_id, "reason": "missing_text_unit"})
                continue

            chunk_text = str(row["text"])
            span_start, span_end = find_chunk_span(raw_document_text, chunk_text, raw_search_start)
            pic_positions = raw_pic_positions
            if span_start != -1:
                raw_search_start = max(raw_search_start + 1, span_start + 1)
                full_text = raw_document_text
            else:
                span_start, span_end = find_chunk_span(source_text, chunk_text, processed_search_start)
                pic_positions = processed_pic_positions
                full_text = source_text
                if span_start != -1:
                    processed_search_start = max(processed_search_start + 1, span_start + 1)

            if span_start == -1:
                unresolved.append(
                    {
                        "document": title,
                        "text_unit_id": text_unit_id,
                        "short_id": str(row["human_readable_id"]),
                        "reason": "span_not_found",
                    }
                )
                pics = []
            else:
                pics = []
                for pic in pic_positions:
                    if span_start <= pic["pos"] < span_end:
                        pics.append(
                            {
                                "pos_in_chunk": pic["pos"] - span_start,
                                "image_id": pic["image_id"],
                                "context": build_pic_context(chunk_text, pic["pos"] - span_start),
                            }
                        )

            item = {
                "text_unit_id": text_unit_id,
                "short_id": str(row["human_readable_id"]),
                "document_title": title,
                "manual_name": manual_name,
                "pics": pics,
            }
            by_text_unit_id[text_unit_id] = item
            by_short_id[str(row["human_readable_id"])] = item

    payload = {
        "by_text_unit_id": by_text_unit_id,
        "by_short_id": by_short_id,
        "unresolved": unresolved,
    }
    MAP_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    resolved_with_pics = sum(1 for item in by_text_unit_id.values() if item["pics"])
    print(f"text units mapped: {len(by_text_unit_id)}")
    print(f"text units with pics: {resolved_with_pics}")
    print(f"unresolved: {len(unresolved)}")
    print(f"output: {MAP_FILE}")


if __name__ == "__main__":
    main()
