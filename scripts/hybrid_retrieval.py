"""Add meaning-based search to the existing BM25 passage index.

The embedding file is reusable across Colab sessions. It never contains an API key.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipeline import CorpusSearch


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
CANDIDATES_PER_METHOD = 20


class LocalEmbeddingClient:
    """Use the small local embedding model from the course's retrieval lesson."""

    def __init__(self, model: str = EMBEDDING_MODEL, encoder: Any = None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model)
        self.model = model
        self.encoder = encoder
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> Any:
        if model != self.model:
            raise ValueError(f"Local embedding client expects {self.model}")
        vectors = self.encoder.encode(input, normalize_embeddings=True, show_progress_bar=False)
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=[float(value) for value in vector])
                  for index, vector in enumerate(vectors)],
            usage=SimpleNamespace(total_tokens=0),
        )


def _index_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit_vector(values: list[float]) -> list[float]:
    vector = [float(value) for value in values]
    length = math.sqrt(sum(value * value for value in vector))
    if not vector or not math.isfinite(length) or length == 0:
        raise ValueError("Embedding response contains an empty or invalid vector")
    return [value / length for value in vector]


def _read_embedding_file(path: Path, expected_hash: str, model: str,
                         chunks: list[dict[str, Any]]) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            raise ValueError("Embedding file is empty")
        header = json.loads(first)
        if header.get("index_sha256") != expected_hash or header.get("model") != model:
            raise ValueError("Embedding file does not match the passage index or model; use a new file")
        vectors = {}
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        dimension = None
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            chunk = chunks_by_id.get(item.get("chunk_id"))
            if chunk is None or item.get("content_hash") != chunk["content_hash"]:
                raise ValueError("Embedding file contains a passage that does not match the index")
            if item["chunk_id"] in vectors:
                raise ValueError("Embedding file contains a duplicate passage ID")
            vector = _unit_vector(item["embedding"])
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("Embedding file contains inconsistent vector sizes")
            vectors[item["chunk_id"]] = vector
    return vectors


def build_embedding_index(chunks_path: str | Path, embedding_path: str | Path,
                          client: Any, model: str = EMBEDDING_MODEL,
                          batch_size: int = 32) -> dict[str, Any]:
    """Embed missing passages and resume safely after an interrupted run."""
    if batch_size < 1 or batch_size > 128:
        raise ValueError("batch_size must be between 1 and 128")
    chunks_path = Path(chunks_path)
    embedding_path = Path(embedding_path)
    chunks = CorpusSearch(chunks_path).chunks
    index_sha256 = _index_hash(chunks_path)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    if embedding_path.exists():
        vectors = _read_embedding_file(embedding_path, index_sha256, model, chunks)
    else:
        embedding_path.write_text(json.dumps({"index_sha256": index_sha256, "model": model}) + "\n",
                                  encoding="utf-8")
        vectors = {}
    missing = [chunk for chunk in chunks if chunk["chunk_id"] not in vectors]
    input_tokens_this_run = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        response = client.embeddings.create(model=model, input=[chunk["text"] for chunk in batch])
        input_tokens_this_run += getattr(getattr(response, "usage", None), "total_tokens", 0)
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch) or [item.index for item in ordered] != list(range(len(batch))):
            raise ValueError("Embedding response has missing or duplicate items")
        new_items = []
        for chunk, result in zip(batch, ordered):
            vector = _unit_vector(result.embedding)
            if vectors and len(vector) != len(next(iter(vectors.values()))):
                raise ValueError("Embedding response has a different vector size")
            new_items.append({"chunk_id": chunk["chunk_id"], "content_hash": chunk["content_hash"],
                              "embedding": vector})
        with embedding_path.open("a", encoding="utf-8") as handle:
            for item in new_items:
                handle.write(json.dumps(item, separators=(",", ":")) + "\n")
        vectors.update({item["chunk_id"]: item["embedding"] for item in new_items})
    return {"model": model, "index_sha256": index_sha256, "chunk_count": len(chunks),
            "reused_embeddings": len(chunks) - len(missing), "new_embeddings": len(missing),
            "input_tokens_this_run": input_tokens_this_run,
            "path": str(embedding_path)}


class HybridSearch(CorpusSearch):
    """Dense or BM25+dense search over the same passages and citation IDs."""

    def __init__(self, chunks_path: str | Path, embedding_path: str | Path,
                 client: Any, model: str = EMBEDDING_MODEL):
        super().__init__(chunks_path)
        self.client = client
        self.model = model
        self.vectors = _read_embedding_file(Path(embedding_path), _index_hash(Path(chunks_path)),
                                                 model, self.chunks)
        if len(self.vectors) != len(self.chunks):
            raise ValueError("Embedding file is incomplete; run build_embedding_index first")
        self.query_vectors: dict[str, list[float]] = {}
        self.query_embedding_calls = 0
        self.query_embedding_tokens = 0

    def dense_search(self, query: str, filters: dict[str, str] | None = None,
                     top_k: int = 5) -> list[dict[str, Any]]:
        """Rank matching passages by sentence-transformer cosine similarity."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        filters = filters or {}
        eligible = [chunk for chunk in self.chunks if self._matches_filters(chunk, filters)]
        if not eligible:
            return []
        if query not in self.query_vectors:
            response = self.client.embeddings.create(model=self.model, input=[query])
            self.query_embedding_calls += 1
            self.query_embedding_tokens += getattr(getattr(response, "usage", None), "total_tokens", 0)
            self.query_vectors[query] = _unit_vector(response.data[0].embedding)
        query_vector = self.query_vectors[query]
        if len(query_vector) != len(next(iter(self.vectors.values()))):
            raise ValueError("Query embedding size does not match saved passage embeddings")
        ranked = sorted(
            eligible,
            key=lambda chunk: (-sum(a * b for a, b in zip(query_vector, self.vectors[chunk["chunk_id"]])),
                               chunk["chunk_id"]),
        )[:top_k]
        return [{"rank": rank, "score": round(sum(a * b for a, b in zip(query_vector, self.vectors[chunk["chunk_id"]])), 6),
                 **chunk, "retrieval_method": "dense_cosine"}
                for rank, chunk in enumerate(ranked, 1)]

    def document_search(self, query: str, filters: dict[str, str] | None = None,
                        top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        filters = filters or {}
        # Both methods search the same filtered collection before ranks are fused.
        eligible = [chunk for chunk in self.chunks if self._matches_filters(chunk, filters)]
        if not eligible:
            return []
        bm25 = super().document_search(query, filters, top_k=CANDIDATES_PER_METHOD)
        semantic = self.dense_search(query, filters, top_k=CANDIDATES_PER_METHOD)
        bm25_ranks = {hit["chunk_id"]: rank for rank, hit in enumerate(bm25, 1)}
        semantic_ranks = {hit["chunk_id"]: rank for rank, hit in enumerate(semantic, 1)}
        candidates = set(bm25_ranks) | set(semantic_ranks)
        ranked = sorted(
            candidates,
            key=lambda chunk_id: (
                -(1 / (RRF_K + bm25_ranks[chunk_id]) if chunk_id in bm25_ranks else 0)
                - (1 / (RRF_K + semantic_ranks[chunk_id]) if chunk_id in semantic_ranks else 0),
                chunk_id,
            ),
        )
        results = []
        for rank, chunk_id in enumerate(ranked[:top_k], 1):
            score = ((1 / (RRF_K + bm25_ranks[chunk_id])) if chunk_id in bm25_ranks else 0)
            score += ((1 / (RRF_K + semantic_ranks[chunk_id])) if chunk_id in semantic_ranks else 0)
            results.append({"rank": rank, "score": round(score, 6), **self.by_id[chunk_id],
                            "retrieval_method": "hybrid_rrf", "bm25_rank": bm25_ranks.get(chunk_id),
                            "semantic_rank": semantic_ranks.get(chunk_id)})
        return results

    def diverse_search(self, query: str, filters: dict[str, str] | None = None,
                       top_k: int = 5) -> list[dict[str, Any]]:
        """Show one passage per page or report for a quick first look.

        Use document_search for a deeper look at nearby passages on the same page.
        """
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        candidates = self.document_search(query, filters, top_k=20)
        selected = []
        seen_locators = set()
        for hit in candidates:
            key = (hit["source_id"], hit["locator"])
            if key in seen_locators:
                continue
            seen_locators.add(key)
            selected.append({**hit, "rank": len(selected) + 1, "retrieval_method": "hybrid_rrf_diverse"})
            if len(selected) == top_k:
                break
        return selected
