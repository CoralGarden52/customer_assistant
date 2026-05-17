"""Unified submission: all 400 questions with automatic language detection."""
import asyncio
import csv
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PARENT_ROOT))

from graphrag.config.load_config import load_config
from graphrag.utils.storage import load_table_from_storage
from graphrag.utils.api import create_storage_from_config
from graphrag_common import BM25Index, process_one, graphrag_search, llm_answer
from image_utils import get_image_mapper

QUESTION_CSV  = PARENT_ROOT / "data" / "question_public.csv"
OUTPUT_CSV    = PROJECT_ROOT / "submission.csv"
PROMPT_FILE   = PROJECT_ROOT / "prompts" / "customer_service_system_prompt.txt"

SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")
CONCURRENCY = 3

FALLBACK_CN = [
    "您好，您的问题已收到，请您耐心等待处理结果，谢谢。",
    "好的，没问题的，我们会尽快给您答复的。",
    "正在为您核实，由于咨询人数较多，请您稍后再来咨询。",
    "很抱歉给您带来不便，我们已记录您的反馈，祝您生活愉快。",
    "感谢您的咨询，相关问题正在后台核实中，请随时关注后续动态。",
    "收到相关反馈，我们正在加急处理，请稍候。",
]
FALLBACK_EN = [
    "Thank you for your inquiry. We have received your question and will get back to you shortly.",
    "We apologize for the inconvenience. Your feedback has been recorded and we will follow up soon.",
    "Thank you for contacting us. We are looking into this and will respond as soon as possible.",
]


def clean_answer_text(text: str) -> str:
    """Normalize answer text while preserving quotes for CSV escaping."""
    if text is None:
        return ""
    return text.strip()


def format_answer_segment(text: str, images: list[str] | None = None) -> str:
    """Format one answer segment as `"answer"` or `"answer", ["img1", "img2"]`."""
    segment = f'"{text}"'
    if images:
        image_str = ", ".join(f'"{img}"' for img in images)
        segment += f", [{image_str}]"
    return segment


def format_multiline_csv_field(segments: list[str]) -> str:
    """Format multi-line content like `"line1"`,\n`"line2"` with optional image lists."""
    return ",\n".join(segment for segment in segments if segment)


def is_english(text: str) -> bool:
    """Detect if text is primarily English. Uses Chinese character count as primary signal."""
    cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
    total_chars = len(text.strip())
    if total_chars == 0:
        return True
    # If more than 20% Chinese characters, treat as Chinese
    return cn_chars / total_chars < 0.2


def fix_language_mismatch(question: str, answer: str, fallbacks: list[str]) -> str:
    """Fix language mismatch between question and answer.

    If question is English but answer is Chinese (or vice versa), return appropriate fallback.
    """
    q_is_english = is_english(question)
    a_is_english = is_english(answer)

    if q_is_english != a_is_english:
        # Language mismatch detected
        return fallbacks[0]  # Return first fallback in appropriate language
    return answer


async def run_all():
    print("=" * 60, flush=True)
    print("Unified Submission Generator (all 400 questions)", flush=True)
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
    texts = text_units_df["text"].tolist()
    bm25_index = BM25Index(text_units_df)
    print(f"  Indexed {len(texts)} text chunks", flush=True)

    print("Loading all questions...", flush=True)
    questions = []
    with open(QUESTION_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = int(row["id"])
            q = row["question"]
            # Preserve multi-turn format: "line1",\n"line2"
            # Only normalize whitespace within each line, don't strip quotes
            lines = q.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1].strip()
                cleaned_lines.append(line)
            q = '\n'.join(cleaned_lines)
            questions.append((qid, q))
    print(f"  {len(questions)} questions", flush=True)

    # Count language distribution
    en_count = sum(1 for _, q in questions if is_english(q))
    print(f"  Chinese: {len(questions) - en_count}, English: {en_count}", flush=True)

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
    rag_count = fb_count = completed = 0
    total = len(remaining)
    lock = asyncio.Lock()

    csv_file = open(OUTPUT_CSV, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["id", "ret"])
    if not done_ids:
        writer.writeheader()

    def parse_multi_turn(question):
        """Parse multi-turn question into individual questions."""
        lines = question.split('\n')
        sub_questions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove trailing comma
            if line.endswith(','):
                line = line[:-1].strip()
            # Remove only wrapper quotes used by the CSV representation
            line = line.strip('"').strip()
            if line:
                sub_questions.append(line)
        return sub_questions if len(sub_questions) > 1 else None

    async def worker(qid, question):
        nonlocal rag_count, fb_count, completed
        # Detect language and use appropriate fallback
        fallbacks = FALLBACK_EN if is_english(question) else FALLBACK_CN
        answer = fallbacks[0]
        is_fb = True
        elapsed = 0
        selected_images = []

        try:
            async with semaphore:
                # Check if multi-turn question
                sub_questions = parse_multi_turn(question)

                if sub_questions:
                    # Multi-turn: process each sub-question separately so each one can
                    # keep its own <PIC> markers and image list.
                    answer_segments = []
                    total_elapsed = 0
                    for sq in sub_questions:
                        _, sq_answer, sq_is_fb, sq_elapsed, sq_images = await process_one(
                            config, dfs, bm25_index, qid, sq, SYSTEM_PROMPT, fallbacks, "unified"
                        )

                        if not sq_is_fb:
                            sq_answer = fix_language_mismatch(sq, sq_answer, fallbacks)
                            if sq_answer == fallbacks[0] and sq_images:
                                sq_images = []

                        sq_answer = clean_answer_text(sq_answer)
                        answer_segments.append(format_answer_segment(sq_answer, sq_images))
                        total_elapsed += sq_elapsed

                    answer = format_multiline_csv_field(answer_segments)

                    is_fb = False
                    elapsed = total_elapsed
                    selected_images = []
                else:
                    # Single question: normal processing
                    qid, answer, is_fb, elapsed, selected_images = await process_one(
                        config, dfs, bm25_index, qid, question, SYSTEM_PROMPT, fallbacks, "unified")

                    # Post-processing: fix any remaining language mismatches
                    if not is_fb:
                        answer = fix_language_mismatch(question, answer, fallbacks)
                        if answer == fallbacks[0] and selected_images:
                            selected_images = []

                    answer = clean_answer_text(answer)
                    answer = format_answer_segment(answer, selected_images)

        except Exception as e:
            import traceback
            print(f"  Error processing qid={qid}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            traceback.print_exc()
            answer = fallbacks[0]
            is_fb = True
            selected_images = []

        async with lock:
            writer.writerow({"id": qid, "ret": answer})
            csv_file.flush()
            completed += 1
            if is_fb:
                fb_count += 1
            else:
                rag_count += 1
            if completed % 10 == 0 or completed == 1:
                lang = "EN" if is_english(question) else "CN"
                img_count = len(selected_images) if selected_images else 0
                print(f"  [{completed}/{total}] id={qid} [{lang}] ({elapsed:.1f}s) imgs={img_count}", flush=True)

    try:
        await asyncio.gather(*[worker(qid, q) for qid, q in remaining], return_exceptions=True)
    except Exception as e:
        print(f"Error during processing: {e}", flush=True)
    finally:
        csv_file.close()

    print("=" * 60, flush=True)
    print(f"Done! {completed} answers -> {OUTPUT_CSV}", flush=True)
    print(f"  RAG: {rag_count}, Fallback: {fb_count}", flush=True)


def main():
    asyncio.run(run_all())

if __name__ == "__main__":
    main()
