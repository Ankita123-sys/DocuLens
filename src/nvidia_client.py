from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import httpx

from src.config import Settings


class NvidiaClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._headers = {
            "Authorization": f"Bearer {settings.nvidia_api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = httpx.Timeout(
            connect=30.0,
            read=settings.http_timeout_seconds,
            write=60.0,
            pool=30.0,
        )
        self._client = httpx.Client(
            timeout=self._timeout,
            headers=self._headers,
            limits=httpx.Limits(
                max_connections=settings.embed_concurrency + 2,
                max_keepalive_connections=settings.embed_concurrency + 2,
            ),
        )

    def close(self) -> None:
        self._client.close()

    def _request_with_retries(self, url: str, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self._settings.embed_max_retries):
            try:
                response = self._client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as exc:
                last_error = exc
                if attempt + 1 >= self._settings.embed_max_retries:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(
            f"NVIDIA API timed out after {self._settings.embed_max_retries} attempts."
        ) from last_error

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        payload = {
            "model": self._settings.embed_model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        url = f"{self._settings.nvidia_base_url}/embeddings"
        data = self._request_with_retries(url, payload)
        items = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]

    def embed_texts(
        self,
        texts: list[str],
        input_type: str = "passage",
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        batch_size = self._settings.embed_batch_size
        concurrency = self._settings.embed_concurrency
        batch_list = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        ordered: list[list[list[float]] | None] = [None] * len(batch_list)
        total = len(texts)
        done = 0
        lock = threading.Lock()

        def embed_indexed(index: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            return index, self._embed_batch(batch, input_type)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(embed_indexed, index, batch)
                for index, batch in enumerate(batch_list)
            ]
            for future in as_completed(futures):
                index, batch_embeddings = future.result()
                ordered[index] = batch_embeddings
                with lock:
                    done += len(batch_list[index])
                    if on_progress:
                        on_progress(done, total)

        flat: list[list[float]] = []
        for batch in ordered:
            if batch is None:
                raise RuntimeError("Embedding batch failed to complete.")
            flat.extend(batch)
        return flat

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], input_type="query")[0]

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._settings.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": False,
        }
        url = f"{self._settings.nvidia_base_url}/chat/completions"
        data = self._request_with_retries(url, payload)
        return data["choices"][0]["message"]["content"].strip()
