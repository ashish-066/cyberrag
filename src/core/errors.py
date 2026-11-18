"""Typed errors shared by the service, the FastAPI workbench and the MCP server.

Every failure that can reach a client carries a stable ``code``, a human
``message`` and an actionable ``hint`` (PRD section 7.6).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

ERROR_CODES = (
    "INVALID_INPUT",
    "UNKNOWN_RELATION",
    "ENTITY_NOT_FOUND",
    "NO_EVIDENCE",
    "VERIFIER_UNAVAILABLE",
    "LLM_UNAVAILABLE",
    "UPLOAD_REJECTED",
    "NO_EXTRACTABLE_TEXT",
    "TOKEN_EXPIRED",
    "RATE_LIMITED",
    "INTERNAL",
)


class CTIError(Exception):
    """Base class. ``code`` is one of :data:`ERROR_CODES`."""

    code = "INTERNAL"
    default_hint = "Retry; if the problem persists, report the trace_id."

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.hint = hint or self.default_hint
        self.trace_id = trace_id
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
        }
        if self.trace_id:
            body["trace_id"] = self.trace_id
        if self.details:
            body["details"] = self.details
        return body


class InvalidInputError(CTIError):
    code = "INVALID_INPUT"
    default_hint = "Check the parameter limits in the tool description."


class UnknownRelationError(CTIError):
    code = "UNKNOWN_RELATION"
    default_hint = "Use one of USES, TARGETS, EXPLOITS, USES_TECHNIQUE."


class EntityNotFoundError(CTIError):
    code = "ENTITY_NOT_FOUND"
    default_hint = "Try an alias or check the spelling; the corpus is English only."


class NoEvidenceError(CTIError):
    code = "NO_EVIDENCE"
    default_hint = "No passage mentions the entities together."


class VerifierUnavailableError(CTIError):
    code = "VERIFIER_UNAVAILABLE"
    default_hint = (
        "Run scripts/download_models.py once while online, then restart. "
        "No verdict is produced without the NLI model."
    )


class LLMUnavailableError(CTIError):
    code = "LLM_UNAVAILABLE"
    default_hint = "Verdicts do not depend on the LLM; a template brief is used."


class UploadRejectedError(CTIError):
    code = "UPLOAD_REJECTED"
    default_hint = "Upload PDF, TXT, MD or HTML up to 20 MB and 200 pages."


class NoExtractableTextError(CTIError):
    code = "NO_EXTRACTABLE_TEXT"
    default_hint = "The file looks scanned or empty; run OCR first and upload text."


class TokenExpiredError(CTIError):
    code = "TOKEN_EXPIRED"
    default_hint = "Call ingest_report_preview again to get a fresh confirm_token."


class RateLimitedError(CTIError):
    code = "RATE_LIMITED"
    default_hint = "Slow down and retry in a few seconds."


class InternalError(CTIError):
    code = "INTERNAL"


class ConfigError(Exception):
    """Raised at startup for invalid configuration (exit code 2)."""

    exit_code = 2


class DataIntegrityError(Exception):
    """Raised at startup when a required data file is missing or hash-mismatched."""

    exit_code = 3

    def __init__(self, message: str, path: Optional[str] = None):
        super().__init__(message)
        self.path = path
