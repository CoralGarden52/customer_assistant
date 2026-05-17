"""Diagnostic script to identify language mismatch issues."""
import asyncio
import csv
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PARENT_ROOT))

from graphrag.config.load_config import load_config
from graphrag.utils.storage import load_table_from_storage
from graphrag.utils.api import create_storage_from_config
from graphrag_common import BM25Index, graphrag_search, is_chinese_text

QUESTION_CSV = PARENT_ROOT / "data" / "question_public.csv"


async def diagnose():
    print("=" * 60)
    print("Language Diagnostic Tool")
    print("=" * 60)

    config = load_config(PROJECT_ROOT, None, {})

    print("Loading indexed data...")
    storage = create_storage_from_config(config.output)
    dfs = {}
    for name in ["entities", "communities", "community_reports", "text_units", "relationships"]:
        try:
            dfs[name] = await load_table_from_storage(name, storage)
        except Exception:
            dfs[name] = pd.DataFrame()

    print("Building BM25 index...")
    text_units_df = dfs["text_units"]
    bm25_index = BM25Index(text_units_df)

    print("Loading questions...")
    questions = []
    with open(QUESTION_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = int(row["id"])
            q = row["question"]
            q = re.sub(r'["""]', '', q)
            q = re.sub(r'\s+', ' ', q).strip()
            questions.append((qid, q))

    # Find English questions
    en_questions = [(qid, q) for qid, q in questions if not is_chinese_text(q)]
    print(f"\nFound {len(en_questions)} English questions")
    print("=" * 60)

    # Test a sample of English questions
    sample_size = min(20, len(en_questions))
    sample_questions = en_questions[:sample_size]

    issues = []
    for qid, question in sample_questions:
        print(f"\n[Q{qid}] {question[:80]}...")

        # Check BM25 results
        bm25_hits = bm25_index.search(question, top_k=3)
        if bm25_hits:
            for i, hit in enumerate(bm25_hits[:2]):
                hit_lang = "CN" if is_chinese_text(hit) else "EN"
                print(f"  BM25 hit {i+1} [{hit_lang}]: {hit[:60]}...")
                if is_chinese_text(hit):
                    issues.append((qid, question, "BM25 returned Chinese chunk"))

        # Check GraphRAG results
        graphrag_ctx = await graphrag_search(config, dfs, question)
        if graphrag_ctx:
            ctx_lang = "CN" if is_chinese_text(graphrag_ctx) else "EN"
            print(f"  GraphRAG [{ctx_lang}]: {graphrag_ctx[:60]}...")
            if is_chinese_text(graphrag_ctx):
                issues.append((qid, question, "GraphRAG returned Chinese context"))

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    if issues:
        print(f"\nFound {len(issues)} issues:")
        for qid, question, issue in issues:
            print(f"  Q{qid}: {issue}")
            print(f"    Question: {question[:60]}...")
    else:
        print("\nNo issues found in retrieval. The language mismatch may be caused by:")
        print("  1. LLM ignoring language instructions")
        print("  2. Bilingual context headers confusing the model")
        print("  3. Model defaulting to Chinese due to training data")

    print("\nRecommendations:")
    print("  1. Run 'python submit.py' to test with the new fixes")
    print("  2. If issues persist, check the LLM model configuration")
    print("  3. Consider adding post-processing to force language match")


def main():
    asyncio.run(diagnose())


if __name__ == "__main__":
    main()
