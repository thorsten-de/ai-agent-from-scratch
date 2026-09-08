"""RAG functionality: embeddings, chunking, and vector search."""

from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_embeddings(texts, model="text-embedding-3-small") -> np.ndarray:
    """Convert text to embedding vectors."""
    client = OpenAI()
    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])


def fixed_length_chunking(text, chunk_size=500, overlap=50) -> list[str]:
    """Split text into fixed-length chunks."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end

    return chunks


def vector_search(query, chunks, chunk_embeddings, top_k=3) -> list:
    """Find the most similar chunks to the query."""
    query_embedding = get_embeddings(query)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'similarity': similarities[idx],
        })
    return results
