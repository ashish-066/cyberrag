"""MCP resources: cti://corpus/stats, cti://report/{report_id}, cti://trace/{verification_id}."""
from __future__ import annotations

import json
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer

from src.core.errors import CTIError, InvalidInputError
from src.core.models import CONTENT_NOTICE


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def register_resources(server: MCPServer, get_service: Callable[[], Any], max_passage_chars: int = 2000) -> None:
    @server.resource(
        "cti://corpus/stats", name="corpus_stats", mime_type="application/json",
        description="Corpus, graph and model statistics plus the data manifest hash.",
    )
    def corpus_stats() -> str:
        return _dump({**get_service().stats(), "content_notice": CONTENT_NOTICE})

    @server.resource(
        "cti://report/{report_id}", name="report", mime_type="application/json",
        description="Report metadata and its segments (segment text is in untrusted_text fields).",
    )
    def report(report_id: str) -> str:
        svc = get_service()
        docs = [d for d in svc.retriever.dense.documents if d.get("report_id") == report_id]
        if not docs:
            raise InvalidInputError(f"Unknown report_id {report_id!r}")
        first = docs[0]
        return _dump(
            {
                "report_id": report_id,
                "publisher": first.get("source"),
                "pub_date": first.get("publication_date"),
                "segment_count": len(docs),
                "segments": [
                    {"segment_id": d.get("segment_id"),
                     "untrusted_text": str(d.get("text") or d.get("clean_text") or "")[:max_passage_chars]}
                    for d in docs[:200]
                ],
                "truncated": len(docs) > 200,
                "content_notice": CONTENT_NOTICE,
            }
        )

    @server.resource(
        "cti://trace/{verification_id}", name="trace", mime_type="application/json",
        description="Full step trace of a verification (retrieval, gate, dedupe, date filter, NLI).",
    )
    def trace(verification_id: str) -> str:
        try:
            return _dump({**get_service().get_trace(verification_id), "content_notice": CONTENT_NOTICE})
        except CTIError:
            raise
