"""Shared utilities for GraphRAG submission scripts."""
import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import jieba
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

CHUNK_META_FILE = Path(__file__).parent / "graphrag_unified" / "chunk_metadata.json"
TEXT_UNIT_IMAGE_MAP_FILE = Path(__file__).parent / "graphrag_unified" / "text_unit_image_map.json"

_stop_words = set(stopwords.words('english'))

# Load chunk metadata for image mapping
_chunk_metadata = None
_text_unit_image_map = None


@dataclass
class RetrievalHit:
    """Structured retrieval unit that keeps text and its linked images together."""

    text: str
    images: list[dict]
    text_unit_id: str | None = None
    short_id: str | None = None


def load_chunk_metadata():
    """Load chunk metadata from JSON file."""
    global _chunk_metadata
    if _chunk_metadata is None and CHUNK_META_FILE.exists():
        with open(CHUNK_META_FILE, encoding="utf-8") as f:
            _chunk_metadata = json.load(f)
    return _chunk_metadata or {}


def load_text_unit_image_map() -> dict:
    """Load stable text_unit_id -> images mapping generated from current index outputs."""
    global _text_unit_image_map
    if _text_unit_image_map is None and TEXT_UNIT_IMAGE_MAP_FILE.exists():
        with open(TEXT_UNIT_IMAGE_MAP_FILE, encoding="utf-8") as f:
            _text_unit_image_map = json.load(f)
    return _text_unit_image_map or {}


def normalize_chunk_text(text: str) -> str:
    """Normalize chunk text for robust metadata matching."""
    if not text:
        return ""
    text = text.replace("\\u201c", '"').replace("\\u201d", '"')
    text = text.replace("\\u2018", "'").replace("\\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_text_unit_images(
    text_unit_id: str | None = None,
    short_id: str | None = None,
    chunk_text: str | None = None,
):
    """Get images for a text unit using stable ID mapping, with text fallback.

    Args:
        text_unit_id: Full text unit ID from text_units.parquet
        short_id: Short ID used by GraphRAG context tables
        chunk_text: Chunk text used only as a fallback when no stable ID is available

    Returns:
        List of dicts with image_id and context
    """
    mapping = load_text_unit_image_map()
    if mapping:
        if text_unit_id:
            item = mapping.get("by_text_unit_id", {}).get(str(text_unit_id))
            if item:
                return item.get("pics", [])
        if short_id is not None:
            item = mapping.get("by_short_id", {}).get(str(short_id))
            if item:
                return item.get("pics", [])

    if chunk_text is None:
        return []

    metadata = load_chunk_metadata()
    if not metadata:
        return []

    normalized_query = normalize_chunk_text(chunk_text)
    if not normalized_query:
        return []

    exact_match = None
    fuzzy_match = None

    # Search through metadata to find matching chunk
    for mname, mdata in metadata.items():
        for chunk in mdata.get("chunks", []):
            metadata_text = chunk.get("text", "")
            if not metadata_text:
                continue

            # Fast path: preserve the old exact prefix behavior.
            if metadata_text[:100] == chunk_text[:100]:
                exact_match = chunk.get("pics", [])
                if exact_match:
                    return exact_match

            normalized_meta = normalize_chunk_text(metadata_text)
            if not normalized_meta:
                continue

            query_prefix = normalized_query[:120]
            meta_prefix = normalized_meta[:120]

            # Fallback for English/escaped chunks whose split point differs between
            # text_units and chunk_metadata: allow normalized overlap matching.
            if query_prefix and (query_prefix in normalized_meta or meta_prefix in normalized_query):
                if chunk.get("pics", []):
                    fuzzy_match = chunk.get("pics", [])
                    return fuzzy_match

    return fuzzy_match or exact_match or []


def get_chunk_images(chunk_text):
    """Backward-compatible wrapper for text-based image lookup."""
    return get_text_unit_images(chunk_text=chunk_text)


def get_images_for_chunks(chunks):
    """Get images for multiple chunks.

    Args:
        chunks: List of chunk texts

    Returns:
        List of lists of image dicts
    """
    return [get_chunk_images(chunk) for chunk in chunks]


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize mixed Chinese+English text: jieba for Chinese, NLTK for English."""
    tokens = []
    for segment in re.split(r'([一-鿿]+)', text):
        if re.match(r'[一-鿿]+', segment):
            tokens.extend(jieba.lcut(segment))
        elif segment.strip():
            words = word_tokenize(segment.lower())
            words = [w for w in words if w.isalpha() and w not in _stop_words and len(w) > 1]
            tokens.extend(words)
    return tokens


def tokenize_chinese(text: str) -> list[str]:
    """Tokenize Chinese text only."""
    return [t for t in jieba.lcut(text) if t.strip()]


def tokenize_english(text: str) -> list[str]:
    """Tokenize English text only."""
    words = word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in _stop_words and len(w) > 1]


def is_chinese_text(text: str) -> bool:
    """Check if text is primarily Chinese."""
    cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
    return cn_chars / max(len(text), 1) > 0.3


class BM25Index:
    """Dual BM25 index: separate Chinese and English indexes for language-aware retrieval."""

    def __init__(self, texts: pd.DataFrame | list[str]):
        if isinstance(texts, pd.DataFrame):
            self.texts = texts["text"].tolist()
            self.text_unit_ids = texts["id"].astype(str).tolist()
            self.short_ids = texts["human_readable_id"].astype(str).tolist()
        else:
            self.texts = texts
            self.text_unit_ids = [None] * len(texts)
            self.short_ids = [str(i) for i in range(len(texts))]

        # Split texts by language
        self.cn_indices = []
        self.en_indices = []
        cn_tokenized = []
        en_tokenized = []

        for i, text in enumerate(self.texts):
            if is_chinese_text(text):
                self.cn_indices.append(i)
                cn_tokenized.append(tokenize_chinese(text))
            else:
                self.en_indices.append(i)
                en_tokenized.append(tokenize_english(text))

        self.bm25_cn = BM25Okapi(cn_tokenized) if cn_tokenized else None
        self.bm25_en = BM25Okapi(en_tokenized) if en_tokenized else None
        print(f"  BM25 index: {len(self.cn_indices)} CN chunks, {len(self.en_indices)} EN chunks", flush=True)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Search in the appropriate language index only (no cross-language to avoid language mixing)."""
        return [hit.text for hit in self.search_hits(query, top_k=top_k)]

    def search_hits(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Search and return structured hits with linked image metadata."""
        results = []
        seen = set()

        if is_chinese_text(query):
            # Chinese query: search Chinese index only
            if self.bm25_cn:
                tokens = tokenize_chinese(query)
                scores = self.bm25_cn.get_scores(tokens)
                for idx in scores.argsort()[-top_k:][::-1]:
                    if scores[idx] > 0:
                        real_idx = self.cn_indices[idx]
                        results.append(
                            RetrievalHit(
                                text=self.texts[real_idx],
                                images=get_text_unit_images(
                                    text_unit_id=self.text_unit_ids[real_idx],
                                    short_id=self.short_ids[real_idx],
                                    chunk_text=self.texts[real_idx],
                                ),
                                text_unit_id=self.text_unit_ids[real_idx],
                                short_id=self.short_ids[real_idx],
                            )
                        )
                        seen.add(real_idx)
        else:
            # English query: search English index only
            if self.bm25_en:
                tokens = tokenize_english(query)
                scores = self.bm25_en.get_scores(tokens)
                for idx in scores.argsort()[-top_k:][::-1]:
                    if scores[idx] > 0:
                        real_idx = self.en_indices[idx]
                        results.append(
                            RetrievalHit(
                                text=self.texts[real_idx],
                                images=get_text_unit_images(
                                    text_unit_id=self.text_unit_ids[real_idx],
                                    short_id=self.short_ids[real_idx],
                                    chunk_text=self.texts[real_idx],
                                ),
                                text_unit_id=self.text_unit_ids[real_idx],
                                short_id=self.short_ids[real_idx],
                            )
                        )
                        seen.add(real_idx)

        return results


async def graphrag_search(config, dfs, question: str) -> str:
    response, _ = await graphrag_search_with_hits(config, dfs, question)
    return response


def extract_graphrag_text_unit_hits(context_data: object) -> list[RetrievalHit]:
    """Convert GraphRAG local-search context_data into retrieval hits with images."""
    if not isinstance(context_data, dict):
        return []

    text_units_df = context_data.get("text_units")
    if not isinstance(text_units_df, pd.DataFrame) or text_units_df.empty:
        return []

    hits = []
    seen = set()
    for _, row in text_units_df.iterrows():
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        normalized = normalize_chunk_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        short_id = str(row.get("id", "")).strip() or None
        hits.append(
            RetrievalHit(
                text=text,
                images=get_text_unit_images(short_id=short_id, chunk_text=text),
                short_id=short_id,
            )
        )
    return hits


async def graphrag_search_with_hits(
    config, dfs, question: str
) -> tuple[str, list[RetrievalHit]]:
    import graphrag.api as api
    try:
        response, context_data = await api.local_search(
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
        response_text = response if isinstance(response, str) else str(response)
        return response_text, extract_graphrag_text_unit_hits(context_data)
    except Exception:
        return "", []


def deduplicate_context(bm25_chunks: list[str], graphrag_context: str) -> tuple[str, str]:
    """Remove duplicate content between BM25 and GraphRAG results."""
    if not bm25_chunks or not graphrag_context:
        bm25_ctx = "\n---\n".join(bm25_chunks) if bm25_chunks else ""
        return bm25_ctx, graphrag_context

    # Find BM25 chunks that are already in GraphRAG context
    unique_bm25 = []
    for chunk in bm25_chunks:
        # Use first 100 chars as fingerprint to check overlap
        fingerprint = chunk[:100].strip()
        if fingerprint and fingerprint not in graphrag_context:
            unique_bm25.append(chunk)

    bm25_ctx = "\n---\n".join(unique_bm25) if unique_bm25 else ""
    return bm25_ctx, graphrag_context


def dedupe_image_candidates(images: list[dict]) -> list[dict]:
    """Keep image candidates in source order while removing duplicates."""
    result = []
    seen = set()
    for img in images:
        image_id = img.get("image_id")
        if image_id and image_id not in seen:
            result.append(img)
            seen.add(image_id)
    return result


async def llm_answer(config, question: str, bm25_context: str, graphrag_context: str,
                     system_prompt: str, label: str = "answer") -> str:
    from graphrag.language_model.providers.litellm.chat_model import LitellmChatModel

    model_cfg = config.models["default_chat_model"]
    model = LitellmChatModel(name=label, config=model_cfg)

    # Deduplicate context
    bm25_ctx, graphrag_ctx = deduplicate_context(
        bm25_context.split("\n---\n") if bm25_context else [],
        graphrag_context
    )

    # Use language-aware section headers
    q_is_cn = is_chinese_text(question)
    if q_is_cn:
        header_bm25 = "【关键词检索结果】"
        header_graphrag = "【语义检索结果】"
        fallback_text = "未找到相关内容。"
    else:
        header_bm25 = "Keyword Search Results"
        header_graphrag = "Semantic Search Results"
        fallback_text = "No relevant content found."

    context_parts = []
    if bm25_ctx:
        context_parts.append(f"{header_bm25}\n{bm25_ctx}")
    if graphrag_ctx:
        context_parts.append(f"{header_graphrag}\n{graphrag_ctx}")

    context = "\n\n".join(context_parts) if context_parts else fallback_text
    # Escape curly braces in context and question to prevent format errors
    safe_context = context.replace("{", "{{").replace("}", "}}")
    safe_question = question.replace("{", "{{").replace("}", "}}")
    prompt = system_prompt.format(context=safe_context, question=safe_question)

    try:
        resp = await model.achat(prompt)
        return resp.output.content.strip()
    except Exception as e:
        print(f"    LLM error: {e}", flush=True)
        return ""


async def select_relevant_images(config, question, chunks_with_images, label="image_select"):
    """Use LLM to select relevant images from retrieved chunks.

    Args:
        config: GraphRAG config
        question: User's question
        chunks_with_images: List of (chunk_text, images) tuples

    Returns:
        List of relevant image IDs (in order matching <PIC> markers)
    """
    from graphrag.language_model.providers.litellm.chat_model import LitellmChatModel

    # Collect all images from chunks, preserving order
    all_images = []
    seen_ids = set()
    for chunk_text, images in chunks_with_images:
        for img in images:
            if img.get("image_id") and img["image_id"] not in seen_ids:
                all_images.append(img)
                seen_ids.add(img["image_id"])

    if not all_images:
        return []

    # If only a few images, return all (preserving order)
    if len(all_images) <= 2:
        return [img["image_id"] for img in all_images]

    # Build prompt for LLM selection
    image_info = []
    for i, img in enumerate(all_images, 1):
        ctx = img.get("context", "").replace("\n", " ")[:80]
        image_info.append(f"{i}. {img['image_id']}: {ctx}")

    # Escape curly braces in question to prevent f-string issues
    safe_question = question.replace("{", "{{").replace("}", "}}")
    prompt = f"""用户问题: {safe_question}

检索到的图片及上下文:
{chr(10).join(image_info)}

请仔细判断哪些图片与用户问题直接相关。
- 只选择能直接回答问题的图片
- 如果图片内容与问题无关，不要选择
- 宁可少选，不要多选不相关的图片

返回格式：JSON数组，包含相关图片的ID，按它们在原文中出现的顺序排列。
如果没有相关图片，返回空数组 []。

示例输出: ["drill0_08", "drill0_09"] 或 []"""

    try:
        model_cfg = config.models["default_chat_model"]
        model = LitellmChatModel(name=label, config=model_cfg)
        resp = await model.achat(prompt)
        response = resp.output.content.strip()

        # Parse JSON array from response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            selected = json.loads(match.group())
            valid_ids = {img["image_id"] for img in all_images}
            # Filter and preserve order from selection
            result = [img_id for img_id in selected if img_id in valid_ids]
            if result:
                return result
    except Exception as e:
        print(f"    Image selection error: {e}", flush=True)

    # Fallback: return all images in original order
    return [img["image_id"] for img in all_images]


async def insert_pic_markers(config, answer, images_with_context, label="pic_insert"):
    """Insert <PIC> markers into answer based on image context matching.

    Args:
        config: GraphRAG config
        answer: Generated answer text (without <PIC>)
        images_with_context: List of dicts with image_id and context

    Returns:
        Answer with <PIC> markers inserted at appropriate positions
    """
    from graphrag.language_model.providers.litellm.chat_model import LitellmChatModel

    if not images_with_context:
        return answer, []

    # Build image info for the prompt
    image_info = []
    for i, img in enumerate(images_with_context, 1):
        ctx = img.get("context", "").replace("\n", " ")[:100]
        image_info.append(f"图片{i} ({img['image_id']}): {ctx}")

    # Escape curly braces to prevent f-string issues
    safe_answer = answer.replace("{", "{{").replace("}", "}}")
    prompt = f"""请在以下回答中插入<PIC>标记。

回答内容:
{safe_answer}

可用的图片（按顺序排列）:
{chr(10).join(image_info)}

规则:
1. 保持回答的语言不变（中文回答保持中文，英文回答保持英文）
2. 只插入与回答内容直接相关的图片，不相关的图片不要插入<PIC>
3. 必须保持图片的顺序：图片1的<PIC>必须在图片2的<PIC>前面，图片2的<PIC>必须在图片3的<PIC>前面，以此类推
4. 将<PIC>插入到回答中描述该图片内容的位置后面
5. 保持回答的其他内容不变，不要翻译或改写
6. 只返回插入<PIC>后的回答，不要添加其他内容

示例（正确顺序）:
输入回答: "电池组充电中显示红色，充满后常亮"
图片1: 充电中状态
图片2: 充满状态
输出: "电池组充电中显示红色<PIC>充满后常亮<PIC>"

示例（跳过不相关图片）:
输入回答: "电池组充电中显示红色，充满后常亮"
图片1: 充电中状态 (相关)
图片2: 电池外观 (不相关)
图片3: 充满状态 (相关)
输出: "电池组充电中显示红色<PIC>充满后常亮<PIC>"
注意: 图片2被跳过，但图片1和图片3的顺序保持不变

请返回插入<PIC>后的回答:"""

    try:
        model_cfg = config.models["default_chat_model"]
        model = LitellmChatModel(name=label, config=model_cfg)
        resp = await model.achat(prompt)
        result = resp.output.content.strip()

        # Verify <PIC> count matches image count
        pic_count = result.count("<PIC>")
        image_count = len(images_with_context)

        if pic_count == image_count:
            # Perfect match: all images have corresponding <PIC>
            image_ids = [img["image_id"] for img in images_with_context]
            return result, image_ids
        elif pic_count > 0 and pic_count < image_count:
            # LLM skipped some images (presumably irrelevant)
            # Since we instructed LLM to preserve order, the first pic_count images were selected
            image_ids = [img["image_id"] for img in images_with_context[:pic_count]]
            return result, image_ids
        elif pic_count > image_count:
            # Unexpected: more <PIC> than images - truncate <PIC> markers
            # This shouldn't happen if LLM follows instructions
            image_ids = [img["image_id"] for img in images_with_context]
            return result, image_ids
        else:
            # No <PIC> inserted - LLM determined no images are relevant
            # Return answer WITHOUT images
            return answer, []

    except Exception as e:
        print(f"    PIC insertion error: {e}", flush=True)
        # Fallback
        image_ids = [img["image_id"] for img in images_with_context]
        return answer, image_ids


async def process_one(config, dfs, bm25_index, qid, question, system_prompt, fallbacks, label="answer"):
    t0 = time.time()
    try:
        bm25_hits = bm25_index.search_hits(question, top_k=5)
        bm25_ctx = "\n---\n".join(hit.text for hit in bm25_hits) if bm25_hits else ""
        graphrag_ctx, graphrag_hits = await graphrag_search_with_hits(config, dfs, question)
        retrieval_hits = graphrag_hits + bm25_hits
        image_candidates = dedupe_image_candidates(
            [img for hit in graphrag_hits for img in hit.images]
            + [img for hit in bm25_hits for img in hit.images]
        )
        answer = await llm_answer(
            config,
            question,
            bm25_ctx,
            graphrag_ctx,
            system_prompt,
            label,
        )

        if not answer or len(answer.strip()) < 5:
            answer = fallbacks[qid % len(fallbacks)]
            is_fb = True
            selected_images = []
        else:
            is_fb = False

            # Language safety check: ensure answer language matches question language
            q_is_cn = is_chinese_text(question)
            a_is_cn = is_chinese_text(answer)
            if q_is_cn != a_is_cn:
                # Retry with explicit language instruction
                lang_instruction = "请用中文回答。" if q_is_cn else "Please answer in English."
                context_combined = bm25_ctx + "\n" + graphrag_ctx
                safe_context_retry = context_combined.replace("{", "{{").replace("}", "}}")
                safe_question_retry = question.replace("{", "{{").replace("}", "}}")
                retry_prompt = f"{lang_instruction}\n\n{system_prompt.format(context=safe_context_retry, question=safe_question_retry)}"
                retry_answer = await llm_answer(
                    config,
                    question,
                    bm25_ctx,
                    graphrag_ctx,
                    retry_prompt,
                    label,
                )
                if retry_answer and is_chinese_text(retry_answer) == q_is_cn:
                    answer = retry_answer

            if image_candidates:
                chunks_with_images = [
                    (hit.text, hit.images) for hit in retrieval_hits if hit.images
                ]
                selected_image_ids = await select_relevant_images(
                    config,
                    question,
                    chunks_with_images,
                    label=f"{label}_image_select",
                )
                images_with_context = [
                    img for img in image_candidates if img.get("image_id") in selected_image_ids
                ]
                if images_with_context:
                    answer, selected_images = await insert_pic_markers(
                        config,
                        answer,
                        images_with_context,
                        label=f"{label}_pic_insert",
                    )
                else:
                    selected_images = []
            else:
                selected_images = []

            # Final language check after PIC insertion
            if is_chinese_text(answer) != q_is_cn:
                # Retry without images if insertion drifts into the wrong language.
                answer = await llm_answer(config, question, bm25_ctx, graphrag_ctx, system_prompt, label)
                selected_images = []

    except Exception as e:
        import traceback
        print(f"    Error in process_one for qid={qid}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        traceback.print_exc()
        answer = fallbacks[qid % len(fallbacks)]
        is_fb = True
        selected_images = []
        elapsed = time.time() - t0
        return qid, answer, is_fb, elapsed, selected_images

    elapsed = time.time() - t0
    return qid, answer, is_fb, elapsed, selected_images
