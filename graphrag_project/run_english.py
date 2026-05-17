"""
English-only submission: process questions from id 241 onwards.
Uses the same hybrid retrieval but optimized for English.
"""
import asyncio
import csv
import re
import sys
import time
from pathlib import Path

import pandas as pd
import jieba
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.utils.storage import load_table_from_storage
from graphrag.utils.api import create_storage_from_config

QUESTION_CSV  = Path("/home/wyz/kefu/data/question_public.csv")
OUTPUT_CSV    = Path("/home/wyz/kefu/data/submission_english.csv")
PROMPT_FILE   = PROJECT_ROOT / "prompts" / "customer_service_system_prompt.txt"

SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")

CONCURRENCY = 5
ENGLISH_START_ID = 241

_stop_words = set(stopwords.words('english'))

FALLBACK_EN = [
    "Thank you for your inquiry. We have received your question and will get back to you shortly.",
    "We apologize for the inconvenience. Your feedback has been recorded and we will follow up soon.",
    "Thank you for contacting us. We are looking into this and will respond as soon as possible.",
]


def tokenize_mixed(text: str) -> list[str]:
    tokens = []
    for segment in re.split(r'([一-鿿]+)', text):
        if re.match(r'[一-鿿]+', segment):
            tokens.extend(jieba.lcut(segment))
        elif segment.strip():
            words = word_tokenize(segment.lower())
            words = [w for w in words if w.isalpha() and w not in _stop_words and len(w) > 1]
            tokens.extend(words)
    return tokens


class BM25Index:
    def __init__(self, texts: list[str]):
        self.texts = texts
        tokenized = [tokenize_mixed(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        tokens = tokenize_mixed(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[-top_k:][::-1]
        return [self.texts[i] for i in top_idx if scores[i] > 0]


async def graphrag_search(config, dfs, question: str) -> str:
    try:
        response, _ = await api.local_search(
            config=config,
            entities=dfs["entities"],
            communities=dfs["communities"],
            community_reports=dfs["community_reports"],
            text_units=dfs["text_units"],
            relationships=dfs["relationships"],
            covariates=dfs.get("covariates"),
            community_level=2,
            response_type="Single Paragraph",
            query=question,
            verbose=False,
        )
        return response if isinstance(response, str) else str(response)
    except Exception:
        return ""


async def llm_answer(config, question: str, bm25_context: str, graphrag_context: str) -> str:
    from graphrag.language_model.providers.litellm.chat_model import LitellmChatModel

    model_cfg = config.models["default_chat_model"]
    model = LitellmChatModel(name="answer_en", config=model_cfg)

    context_parts = []
    if bm25_context:
        context_parts.append(f"[Keyword Search Results]\n{bm25_context}")
    if graphrag_context:
        context_parts.append(f"[Semantic Search Results]\n{graphrag_context}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant content found."
    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    try:
        resp = await model.achat(prompt)
        return resp.output.content.strip()
    except Exception as e:
        print(f"    LLM error: {e}", flush=True)
        return ""


async def process_one(config, dfs, bm25_index, qid, question):
    t0 = time.time()
    bm25_hits = bm25_index.search(question, top_k=5)
    bm25_ctx = "\n---\n".join(bm25_hits) if bm25_hits else ""
    graphrag_ctx = await graphrag_search(config, dfs, question)
    answer = await llm_answer(config, question, bm25_ctx, graphrag_ctx)

    if not answer or len(answer.strip()) < 5:
        answer = FALLBACK_EN[qid % len(FALLBACK_EN)]
        is_fb = True
    else:
        is_fb = False

    elapsed = time.time() - t0
    return qid, answer, is_fb, elapsed


async def run_all():
    print("=" * 60, flush=True)
    print("English-Only Submission Generator", flush=True)
    print(f"Concurrency: {CONCURRENCY}", flush=True)
    print("=" * 60, flush=True)

    config = load_config(PROJECT_ROOT, None, {})

    print("Loading indexed data...", flush=True)
    storage = create_storage_from_config(config.output)
    dfs = {}
    for name in ["entities", "communities", "community_reports", "text_units", "relationships"]:
        try:
            dfs[name] = await load_table_from_storage(name, storage)
            print(f"  {name}: {len(dfs[name])} rows", flush=True)
        except Exception as e:
            print(f"  {name}: FAILED ({e})", flush=True)
            dfs[name] = pd.DataFrame()
    try:
        dfs["covariates"] = await load_table_from_storage("covariates", storage)
    except Exception:
        dfs["covariates"] = None

    print("Building BM25 index...", flush=True)
    text_units_df = dfs["text_units"]
    bm25_index = BM25Index(text_units_df)
    print(f"  Indexed {len(texts)} text chunks", flush=True)

    print(f"Loading English questions (id >= {ENGLISH_START_ID})...", flush=True)
    questions = []
    with open(QUESTION_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = int(row["id"])
            if qid < ENGLISH_START_ID:
                continue
            q = row["question"]
            q = re.sub(r'[""“”]', '', q)
            q = re.sub(r'\s+', ' ', q).strip()
            questions.append((qid, q))
    print(f"  {len(questions)} English questions", flush=True)

    # Resume
    done_ids = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(int(row["id"]))
        print(f"  Resume: {len(done_ids)} already done", flush=True)

    remaining = [(qid, q) for qid, q in questions if qid not in done_ids]
    print(f"  Remaining: {len(remaining)}", flush=True)

    if not remaining:
        print("All done!", flush=True)
        return

    print(f"Processing -> {OUTPUT_CSV}", flush=True)
    print("-" * 60, flush=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    rag_count = 0
    fb_count = 0
    completed = 0
    total = len(remaining)
    lock = asyncio.Lock()

    csv_file = open(OUTPUT_CSV, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["id", "ret"])
    if not done_ids:
        writer.writeheader()

    async def worker(qid, question):
        nonlocal rag_count, fb_count, completed
        async with semaphore:
            qid, answer, is_fb, elapsed = await process_one(config, dfs, bm25_index, qid, question)
        async with lock:
            writer.writerow({"id": qid, "ret": answer})
            csv_file.flush()
            completed += 1
            if is_fb:
                fb_count += 1
            else:
                rag_count += 1
            if completed % 10 == 0 or completed == 1:
                print(f"  [{completed}/{total}] id={qid} ({elapsed:.1f}s)", flush=True)

    tasks = [worker(qid, q) for qid, q in remaining]
    await asyncio.gather(*tasks)

    csv_file.close()

    print("=" * 60, flush=True)
    print(f"Done! {completed} answers -> {OUTPUT_CSV}", flush=True)
    print(f"  RAG answers: {rag_count}", flush=True)
    print(f"  Fallback: {fb_count}", flush=True)
    print("=" * 60, flush=True)


def main():
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
