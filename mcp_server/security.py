"""MCP security layer: untrusted-text notice, size caps, rate limit, bearer auth, audit log.

Report text only leaves the server in fields named ``untrusted_text`` (S1), and
tool/parameter descriptions are static strings in code (S2).
"""
from __future__ import annotations

import collections
import hmac
import json
import logging
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional

from mcp import types

from src.core.errors import CTIError, InternalError, RateLimitedError
from src.core.logging import args_hash, new_id, trace_id_var
from src.core.models import CONTENT_NOTICE

logger = logging.getLogger("cyberrag.mcp.audit")

_LIST_KEYS = ("results", "supporting", "contradicting", "entries", "neighbors", "entities", "relations")


class RateLimiter:
    """Sliding one-minute window per key."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute = per_minute
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60.0:
                q.popleft()
            if len(q) >= self.per_minute:
                raise RateLimitedError(f"Rate limit of {self.per_minute} calls/minute reached for this session")
            q.append(now)


def _size(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))


def _shrink_texts(node: Any, limit: int) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "untrusted_text" and isinstance(v, str) and len(v) > limit:
                node[k] = v[: limit - 1] + "…"
            else:
                _shrink_texts(v, limit)
    elif isinstance(node, list):
        for v in node:
            _shrink_texts(v, limit)


def enforce_size(payload: Dict[str, Any], max_bytes: int) -> Dict[str, Any]:
    """Cap the response (S5): shorten passage text, then drop trailing list items; flag truncation."""
    if _size(payload) <= max_bytes:
        return payload
    payload["truncated"] = True
    warnings: List[str] = payload.setdefault("warnings", []) if isinstance(payload.get("warnings", []), list) else []
    if "response truncated to fit the size limit" not in warnings:
        warnings.append("response truncated to fit the size limit")
    for limit in (600, 250):
        _shrink_texts(payload, limit)
        if _size(payload) <= max_bytes:
            return payload
    for _ in range(40):
        trimmed = False
        for key in _LIST_KEYS:
            val = payload.get(key)
            if isinstance(val, list) and len(val) > 1:
                del val[max(1, len(val) // 2):]
                trimmed = True
        if _size(payload) <= max_bytes:
            return payload
        if not trimmed:
            break
    payload.pop("bundle", None) if _size(payload) > max_bytes else None
    return payload


def error_result(err: CTIError) -> types.CallToolResult:
    body = err.to_dict()
    text = f"[{body['code']}] {body['message']} Hint: {body['hint']}"
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], structured_content=body, is_error=True
    )


class Guard:
    """Wraps each tool call: rate limit, error mapping, size cap, notice, audit log."""

    def __init__(self, max_response_bytes: int, rate_limiter: Optional[RateLimiter] = None):
        self.max_response_bytes = max_response_bytes
        self.rate_limiter = rate_limiter

    def run(
        self,
        tool: str,
        args: Dict[str, Any],
        fn: Callable[[], Any],
        summarize: Callable[[Dict[str, Any]], str],
        session_id: Optional[str] = None,
    ) -> types.CallToolResult:
        started = time.monotonic()
        outcome = "ok"
        try:
            if self.rate_limiter is not None:
                self.rate_limiter.check(session_id or "anon")
            out = fn()
            payload = out.model_dump(mode="json") if hasattr(out, "model_dump") else dict(out)
            payload.setdefault("content_notice", CONTENT_NOTICE)
            payload = enforce_size(payload, self.max_response_bytes)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=summarize(payload) + "\n" + CONTENT_NOTICE)],
                structured_content=payload,
            )
        except CTIError as exc:
            outcome = exc.code
            exc.trace_id = exc.trace_id or trace_id_var.get()
            return error_result(exc)
        except Exception:  # never leak internals or report text
            tid = new_id("t_")
            logger.exception("tool %s failed (trace %s)", tool, tid)
            outcome = "INTERNAL"
            return error_result(InternalError("Unexpected server error", trace_id=tid))
        finally:
            logger.info(
                "tool_call",
                extra={
                    "tool": tool, "args_hash": args_hash(args), "session": session_id,
                    "duration_ms": round((time.monotonic() - started) * 1000, 1), "outcome": outcome,
                },
            )


class BearerAuthMiddleware:
    """ASGI middleware: constant-time bearer token check for non-local HTTP binds (S6)."""

    def __init__(self, app, token: str):
        self.app = app
        self._token = token.encode("utf-8")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            supplied = headers.get("authorization", "")
            ok = supplied.lower().startswith("bearer ") and hmac.compare_digest(
                supplied[7:].encode("utf-8"), self._token
            )
            if not ok:
                body = json.dumps({"error": {"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token"}}).encode()
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")
