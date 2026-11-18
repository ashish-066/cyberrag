"""Structured JSON logging with request / trace ids carried in contextvars."""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import sys
import time
import uuid
from typing import Any, Optional

request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        body = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "trace_id": trace_id_var.get(),
        }
        for key, val in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                body[key] = val
        if record.exc_info:
            body["exc"] = self.formatException(record.exc_info)
        return json.dumps(body, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_format: bool = True, stream: Any = None) -> None:
    """Idempotent root configuration. MCP stdio servers must log to stderr."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_cti_handler", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler._cti_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        JsonFormatter() if json_format else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


def args_hash(args: Any) -> str:
    """Stable short hash of tool arguments so calls are auditable without logging report text."""
    raw = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
