"""OpenAI-compatible embedding server using local bge-m3 model."""
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("/home/data/wyz/huatuo/models/bge-m3")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "bge-m3"
    encoding_format: str = "float"


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    texts = request.input if isinstance(request.input, list) else [request.input]
    embeddings = model.encode(texts, normalize_embeddings=True)
    data = []
    for i, emb in enumerate(embeddings):
        data.append({
            "object": "embedding",
            "embedding": emb.tolist(),
            "index": i,
        })
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "bge-m3", "object": "model", "owned_by": "local"}],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6100)
