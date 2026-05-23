import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str
    nvidia_base_url: str
    chat_model: str
    embed_model: str
    ld_sdk_key: str
    ld_flag_rag_enabled: str
    ld_user_key: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embed_batch_size: int
    embed_concurrency: int
    http_timeout_seconds: float
    embed_max_retries: int


def get_settings() -> Settings:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not set. Copy .env.example to .env and add your key.")

    ld_key = os.getenv("LAUNCHDARKLY_SDK_KEY", "").strip()
    if not ld_key:
        raise ValueError(
            "LAUNCHDARKLY_SDK_KEY is not set. Copy .env.example to .env and add your key."
        )

    return Settings(
        nvidia_api_key=api_key,
        nvidia_base_url=os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).rstrip("/"),
        chat_model=os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.1-8b-instruct"),
        embed_model=os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embed-v1"),
        ld_sdk_key=ld_key,
        ld_flag_rag_enabled=os.getenv("LD_FLAG_RAG_ENABLED", "rag-enabled"),
        ld_user_key=os.getenv("LD_USER_KEY", "default-user"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "2000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "250")),
        top_k=int(os.getenv("TOP_K", "5")),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "48")),
        embed_concurrency=int(os.getenv("EMBED_CONCURRENCY", "4")),
        http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "300")),
        embed_max_retries=int(os.getenv("EMBED_MAX_RETRIES", "3")),
    )
