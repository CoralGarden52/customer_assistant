"""Query GraphRAG with local reranking using bge-reranker-v2-m3."""
import asyncio
import json
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from sentence_transformers import CrossEncoder

# Load reranker
reranker = CrossEncoder("/home/data/wyz/huatuo/models/bge-reranker-v2-m3", max_length=512)


def rerank(query: str, passages: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """Rerank passages using bge-reranker-v2-m3."""
    pairs = [(query, p) for p in passages]
    scores = reranker.predict(pairs)
    scored = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k]


async def query_graphrag(question: str, method: str = "local"):
    """Run GraphRAG query and rerank results."""
    from graphrag.query import api as query_api

    # Run GraphRAG query
    result = await query_api.global_search(
        root_dir="/home/wyz/kefu/graphrag_project",
        query=question,
    )

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python query_with_rerank.py <question> [method]")
        print("  method: local (default), global, drift, basic")
        sys.exit(1)

    question = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "local"

    print(f"Question: {question}")
    print(f"Method: {method}")
    print("-" * 50)

    result = asyncio.run(query_graphrag(question, method))
    print(result)


if __name__ == "__main__":
    main()
