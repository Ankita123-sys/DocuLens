from __future__ import annotations

import uuid
from typing import Callable

import chromadb

from src.document_processor import DocumentChunk
from src.nvidia_client import NvidiaClient


class SessionVectorStore:
    """In-memory Chroma collection scoped to the Streamlit session."""

    def __init__(self, nvidia: NvidiaClient) -> None:
        self._nvidia = nvidia
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        if self.document_count == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        sources = {
            meta["source"] for meta in result["metadatas"] if meta and "source" in meta
        }
        return sorted(sources)

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> int:
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self._nvidia.embed_texts(
            texts,
            input_type="passage",
            on_progress=on_progress,
        )
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "page": chunk.page if chunk.page is not None else -1,
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int,
        source_filter: str | None = None,
    ) -> list[dict]:
        if self.document_count == 0:
            return []

        query_embedding = self._nvidia.embed_query(question)
        where = {"source": source_filter} if source_filter else None

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.document_count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            page = meta.get("page", -1)
            hits.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", -1),
                    "page": None if page == -1 else int(page),
                    "distance": float(distance),
                }
            )
        return hits
