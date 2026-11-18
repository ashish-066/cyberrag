"""Local LLM client for CyberRAG (LM Studio / llama.cpp / Ollama OpenAI-compatible API).

The LLM is optional everywhere. This client never raises: every failure (server
off, timeout, HTTP error, malformed JSON) returns ``None`` / ``False`` so the
callers fall back to deterministic behaviour. The server URL and model name are
detected lazily from ``/v1/models`` (no hard-coded model label).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PORTS = (1235, 1234, 11434, 8000, 8080)


class LocalLLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_secs: float = 12.0,
        probe_timeout_secs: float = 0.8,
        status_ttl_secs: float = 5.0,
    ):
        self._explicit_url = base_url.rstrip("/") if base_url else None
        self._explicit_model = model
        self.timeout_secs = timeout_secs
        self.probe_timeout_secs = probe_timeout_secs
        self.status_ttl_secs = status_ttl_secs
        self._detected_url: Optional[str] = None
        self._detected_model: Optional[str] = None
        self._checked_at = 0.0
        self._available = False

    # ----------------------------------------------------------- detection
    @property
    def base_url(self) -> str:
        return self._explicit_url or self._detected_url or f"http://127.0.0.1:{DEFAULT_PORTS[0]}/v1"

    @property
    def model(self) -> Optional[str]:
        return self._explicit_model or self._detected_model

    def _probe(self, url: str) -> Optional[List[str]]:
        try:
            req = urllib.request.Request(f"{url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=self.probe_timeout_secs) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("id") for m in data.get("data", []) if m.get("id")]
        except Exception:
            return None

    def refresh(self, force: bool = False) -> bool:
        """Re-probe the server at most every ``status_ttl_secs``."""
        now = time.monotonic()
        if not force and self._checked_at and now - self._checked_at < self.status_ttl_secs:
            return self._available
        urls = [self._explicit_url] if self._explicit_url else [f"http://127.0.0.1:{p}/v1" for p in DEFAULT_PORTS]
        self._available, self._detected_model = False, None
        for url in urls:
            models = self._probe(url)
            if models is not None:
                self._detected_url = url
                self._detected_model = models[0] if models else None
                self._available = bool(models)
                break
        self._checked_at = now
        return self._available

    def is_available(self) -> bool:
        return self.refresh()

    # Back-compat alias: v5 app.py called this name, which did not exist.
    def is_connected(self) -> bool:
        return self.is_available()

    def model_name(self) -> Optional[str]:
        self.refresh()
        return self.model

    # ------------------------------------------------------------- completion
    def chat_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Optional[str]:
        if not self.refresh():
            return None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_secs) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else None
        except Exception as exc:
            logger.warning("Local LLM completion failed (%s): %s", type(exc).__name__, exc)
            self._checked_at = 0.0  # force a re-probe next time
            return None
