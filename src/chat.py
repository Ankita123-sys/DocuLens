from __future__ import annotations

from src.nvidia_client import NvidiaClient

SYSTEM_PROMPT = """You are a helpful document Q&A assistant.
Answer ONLY using the provided context excerpts.
If the context does not contain enough information, say you cannot find it in the documents.
Always end your answer with a "Sources:" section listing each citation as:
- filename (page N) or filename (chunk N) when page is unknown.
Do not invent sources."""


def format_context_block(hits: list[dict]) -> str:
    if not hits:
        return "No relevant excerpts were retrieved."

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        page = hit.get("page")
        page_label = f"page {page}" if page else f"chunk {hit.get('chunk_index', '?')}"
        blocks.append(
            f"[{index}] source={hit['source']} ({page_label})\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def format_citation_preview(hits: list[dict]) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        page = hit.get("page")
        if page:
            label = f"{hit['source']} (p. {page})"
        else:
            label = f"{hit['source']} (chunk {hit.get('chunk_index', '?')})"
        if label not in seen:
            seen.add(label)
            citations.append(label)
    return citations


def answer_question(
    nvidia: NvidiaClient,
    question: str,
    hits: list[dict],
    mode_label: str,
) -> tuple[str, list[str]]:
    context = format_context_block(hits)
    user_prompt = (
        f"Mode: {mode_label}\n\n"
        f"Question:\n{question}\n\n"
        f"Context excerpts:\n{context}\n\n"
        "Provide a concise answer with citations in the Sources section."
    )
    answer = nvidia.chat_completion(SYSTEM_PROMPT, user_prompt)
    citations = format_citation_preview(hits)
    return answer, citations
