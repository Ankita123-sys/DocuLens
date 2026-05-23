# DocuLens

**AI powered Document analyser and query answering app**

DocuLens is a Streamlit app that lets you upload PDF and text files, index them with vector embeddings, and ask natural-language questions with answers grounded in your documents. A **LaunchDarkly** feature flag switches between full-corpus RAG and single-document Q&A.

## Features

- Upload multiple **PDF** and **TXT** files
- **RAG mode** — search across all uploaded documents (flag `rag-enabled` = true)
- **Non-RAG mode** — search only the most recently uploaded file (flag `rag-enabled` = false)
- Answers with **source citations** (filename and page or chunk)
- **NVIDIA NIM** for chat (`meta/llama-3.1-8b-instruct`) and embeddings (`nvidia/nv-embed-v1`)
- **Chroma** in-memory vector store (session-scoped)
- Batched, parallel embedding for faster indexing of large PDFs

## Architecture

```
Upload (PDF/TXT) → chunk → embed (NVIDIA) → Chroma
                              ↓
User question → LaunchDarkly (rag-enabled?) → retrieve chunks → LLM → answer + citations
```

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| LLM | NVIDIA NIM API |
| Embeddings | NVIDIA NIM API |
| Vector DB | Chroma (ephemeral) |
| Feature flags | LaunchDarkly server SDK |

## Prerequisites

- Python 3.10+
- [NVIDIA API key](https://build.nvidia.com) (NIM access)
- [LaunchDarkly](https://launchdarkly.com) account with a **server-side SDK key** (`sdk-...`)
- Boolean flag **`rag-enabled`** in your LaunchDarkly environment

## Quick start

```bash
git clone https://github.com/Ankita123-sys/DocuLens.git
cd DocuLens

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add NVIDIA_API_KEY and LAUNCHDARKLY_SDK_KEY

streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload documents, and ask questions.

## LaunchDarkly setup

1. Create a boolean flag named **`rag-enabled`**.
2. Turn the flag **On** in your target environment (e.g. Test).
3. Set the **default rule** to **Serve `true`** for RAG, or **`false`** for non-RAG.

> **Note:** “Flag is On” only enables targeting rules. The variation served (`true` / `false`) is determined by your rules (e.g. default rule).

4. Copy the **server-side SDK key** (`sdk-...`) for that environment into `.env` as `LAUNCHDARKLY_SDK_KEY`.

Evaluation uses context key `default-user` (override with `LD_USER_KEY`).

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NVIDIA_API_KEY` | Yes | — | NVIDIA NIM API key |
| `LAUNCHDARKLY_SDK_KEY` | Yes | — | LaunchDarkly **server-side** SDK key |
| `NVIDIA_CHAT_MODEL` | No | `meta/llama-3.1-8b-instruct` | Chat model |
| `NVIDIA_EMBED_MODEL` | No | `nvidia/nv-embed-v1` | Embedding model |
| `LD_FLAG_RAG_ENABLED` | No | `rag-enabled` | Feature flag key |
| `LD_USER_KEY` | No | `default-user` | LaunchDarkly context key |
| `CHUNK_SIZE` | No | `2000` | Characters per chunk |
| `CHUNK_OVERLAP` | No | `250` | Chunk overlap |
| `EMBED_BATCH_SIZE` | No | `48` | Texts per embedding API call |
| `EMBED_CONCURRENCY` | No | `4` | Parallel embedding requests |
| `HTTP_TIMEOUT_SECONDS` | No | `300` | API read timeout |
| `TOP_K` | No | `5` | Chunks retrieved per question |

## Large PDFs

Big files are chunked with larger sizes automatically and embedded in parallel batches. If indexing is slow or times out, add to `.env`:

```env
CHUNK_SIZE=3000
EMBED_BATCH_SIZE=64
EMBED_CONCURRENCY=6
```

Lower `EMBED_CONCURRENCY` if you see NVIDIA rate-limit errors (`429`).

## Project structure

```
DocuLens/
├── app.py                 # Streamlit UI
├── requirements.txt
├── .env.example
└── src/
    ├── config.py
    ├── nvidia_client.py
    ├── document_processor.py
    ├── vector_store.py
    ├── launchdarkly_client.py
    └── chat.py
```

## License

MIT
