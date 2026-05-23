from __future__ import annotations

import atexit

import ldclient
from ldclient import Context
from ldclient.config import Config

from src.config import Settings

_client: ldclient.LDClient | None = None


def get_ld_client(settings: Settings) -> ldclient.LDClient:
    global _client
    if _client is None:
        ldclient.set_config(Config(settings.ld_sdk_key))
        _client = ldclient.get()
        if not _client.is_initialized():
            _client.wait_for_initialization(timeout=10)
        atexit.register(_client.close)
    return _client


def is_rag_enabled(settings: Settings) -> bool:
    client = get_ld_client(settings)
    context = Context.builder(settings.ld_user_key).build()
    return bool(client.variation(settings.ld_flag_rag_enabled, context, False))
