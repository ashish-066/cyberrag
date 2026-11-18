"""
CyberRAG CTI Analyst Workbench - FastAPI server.

Thin HTTP layer over ``src.core.service.VerificationService`` (the same service
Streamlit and the MCP server use). The bounded verification workflow plans,
retrieves, gates on entity co-occurrence, runs NLI and resolves conflicts; the
LLM is optional and never decides a verdict.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.retrieval_agent import CyberRAGAgent
from src.core.config import load_settings
from src.core.errors import CTIError, InternalError
from src.core.logging import configure_logging, new_id, request_id_var
from src.core.service import VerificationService

logger = logging.getLogger("cyberrag.server")

STATIC_DIR = Path(__file__).parent / "static"
_SERVICE: Optional[VerificationService] = None
_AGENT: Optional[CyberRAGAgent] = None
_INIT_LOCK = threading.Lock()


def get_service() -> VerificationService:
    """Load models exactly once, even under concurrent first requests (R8)."""
    global _SERVICE, _AGENT
    if _SERVICE is None:
        with _INIT_LOCK:
            if _SERVICE is None:
                settings = load_settings()
                configure_logging(settings.log_level, settings.log_json)
                _SERVICE = VerificationService.from_settings(settings)
                _AGENT = CyberRAGAgent(_SERVICE)
    return _SERVICE


def set_service(service: VerificationService) -> None:
    """Inject a service (tests)."""
    global _SERVICE, _AGENT
    _SERVICE, _AGENT = service, CyberRAGAgent(service)


app = FastAPI(title="CyberRAG CTI Investigation Workbench", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Session-Id"],
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def _request_context(request, call_next):
    rid = request.headers.get("X-Request-Id") or new_id("r_")
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


@app.exception_handler(CTIError)
async def _cti_error(_, exc: CTIError):
    status = {"INVALID_INPUT": 400, "UNKNOWN_RELATION": 400, "UPLOAD_REJECTED": 400,
              "NO_EXTRACTABLE_TEXT": 400, "TOKEN_EXPIRED": 400, "ENTITY_NOT_FOUND": 404,
              "RATE_LIMITED": 429, "VERIFIER_UNAVAILABLE": 503}.get(exc.code, 500)
    return JSONResponse(status_code=status, content={"error": exc.to_dict()})


@app.exception_handler(Exception)
async def _unhandled(_, exc: Exception):
    logger.exception("unhandled error")
    err = InternalError("Unexpected server error", trace_id=request_id_var.get())
    return JSONResponse(status_code=500, content={"error": err.to_dict()})


# ------------------------------------------------------------------ models
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    scenario_id: Optional[str] = None


class ClaimRequest(BaseModel):
    subject: str
    relation: str
    object: str
    as_of_date: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    publishers: Optional[List[str]] = None


class IngestRequest(BaseModel):
    title: str = "Uploaded CTI Report"
    source: str = "Analyst Submission"
    publication_date: str = "2024-01-01"
    text: str


class ConfirmRequest(BaseModel):
    confirm_token: str


# -------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """Liveness: the process is up (does not load models)."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: models loaded. 503 until the service can answer."""
    try:
        svc = get_service()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"ready": False, "problems": [str(exc)]})
    body = svc.ready()
    return JSONResponse(status_code=200 if body["ready"] else 503, content=body)


@app.get("/api/health")
async def api_health():
    svc = get_service()
    st = svc.stats()
    llm = st["llm"]
    return {
        "status": "online",
        "verifier_status": "ready" if st["verifier_ready"] else "failed",
        "verifier_model": st["nli_model"],
        "llm_connected": llm["connected"],
        "active_model": llm["model"] or "Unavailable",
        "active_port": None,
        "indexed_segments": st["segments"],
        "indexed_entities": st["graph_nodes"],
        "indexed_assertions": st["assertions"],
        "mode": st["mode"],
        "demo_overlay_records": st["demo_overlay_records"],
    }


def _subgraph(svc: VerificationService, query: str) -> Dict[str, Any]:
    graph = svc.retriever.graph
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    seen = set()
    for name, eid in graph.match_entities(query):
        nodes[eid] = {"id": eid, "label": name.upper(), "group": "matched_entity",
                      "type": eid.split("::")[0] if "::" in eid else "Entity"}
        for a in graph.entity_index.get(eid, [])[:8]:
            sid, oid = a.get("subject_id", ""), a.get("object_id", "")
            for nid in (sid, oid):
                if nid and nid not in nodes:
                    nodes[nid] = {"id": nid, "label": nid.split("::")[-1], "group": "neighbor",
                                  "type": nid.split("::")[0] if "::" in nid else "Entity"}
            rel = a.get("predicate") or a.get("relation") or "RELATES_TO"
            key = (sid, rel, oid)
            if key not in seen:
                seen.add(key)
                edges.append({"id": f"{sid}->{rel}->{oid}", "from": sid, "to": oid, "label": rel})
    return {"nodes": list(nodes.values()), "edges": edges}


def _legacy_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    nli = item.get("nli") or {}
    return {
        "report_id": item["report_id"], "segment_id": item["segment_id"], "source": item["publisher"],
        "publication_date": item["pub_date"], "text": item["untrusted_text"],
        "source_type": item["source_type"], "synthetic": item["synthetic"],
        "p_entail": nli.get("p_entailment", 0.0), "p_contra": nli.get("p_contradiction", 0.0),
        "p_neutral": nli.get("p_neutral", 0.0), "nli_label": (nli.get("label") or "neutral").upper(),
        "evidence_id": item["evidence_id"],
    }


@app.post("/api/investigate")
async def investigate(req: QueryRequest, x_session_id: Optional[str] = Header(default=None)):
    svc = get_service()
    assert _AGENT is not None
    report = _AGENT.run(req.query.strip(), session_id=x_session_id)
    return {
        "query": report.query,
        "status": report.status,
        "support_tier": report.support_tier,
        "hedged_only": any(c.get("hedged_only") for c in report.claim_results),
        "source_diversity": report.source_diversity,
        "state_history": [s.value for s in report.state_history],
        "execution_trace": [
            {"step_index": s.step_index, "state": s.state.value, "action": s.action, "thought": s.thought,
             "decision_outcome": s.decision_outcome, "retry_or_termination_reason": s.retry_or_termination_reason,
             "tool_name": s.tool_name, "tool_input": s.tool_input, "tool_output_summary": s.tool_output_summary}
            for s in report.execution_trace
        ],
        "supporting_evidence": [_legacy_evidence(e) for e in report.supporting_evidence],
        "contradicting_evidence": [_legacy_evidence(e) for e in report.contradicting_evidence],
        "conflict_resolution": report.conflict_resolution,
        "claims": report.claim_results,
        "plan": report.plan,
        "related_entities": report.related_entities,
        "warnings": report.warnings,
        "narrative": report.narrative,
        "subgraph": _subgraph(svc, req.query),
    }


@app.post("/api/verify_claim")
async def verify_claim(req: ClaimRequest, x_session_id: Optional[str] = Header(default=None)):
    return get_service().verify_claim(
        req.subject, req.relation, req.object, as_of_date=req.as_of_date, session_id=x_session_id
    ).model_dump()


@app.post("/api/search")
async def search(req: SearchRequest, x_session_id: Optional[str] = Header(default=None)):
    return get_service().search(
        req.query, req.top_k, req.date_from, req.date_to, req.publishers, session_id=x_session_id
    ).model_dump()


@app.post("/api/ingest_preview")
async def ingest_preview(req: IngestRequest, x_session_id: Optional[str] = Header(default=None)):
    return get_service().ingest_preview(
        req.text, req.title, req.source, req.publication_date, x_session_id or ""
    ).model_dump()


@app.post("/api/ingest_confirm")
async def ingest_confirm(req: ConfirmRequest, x_session_id: Optional[str] = Header(default=None)):
    return get_service().ingest_confirm(req.confirm_token, x_session_id or "").model_dump()


@app.post("/api/upload_preview")
async def upload_preview(
    file: UploadFile = File(...), title: str = Form("Uploaded CTI Report"),
    source: str = Form("Analyst Submission"), publication_date: str = Form("2024-01-01"),
    x_session_id: Optional[str] = Header(default=None),
):
    svc = get_service()
    data = await file.read(svc.settings.max_upload_bytes + 1)
    return svc.ingest_preview("", title, source, publication_date, x_session_id or "",
                              filename=file.filename, data=data).model_dump()


@app.post("/api/ingest_report")
async def ingest_report(req: IngestRequest, x_session_id: Optional[str] = Header(default=None)):
    """One-click preview + confirm for the workbench; the upload stays in this session only."""
    svc = get_service()
    sid = x_session_id or ""
    prev = svc.ingest_preview(req.text, req.title, req.source, req.publication_date, sid)
    res = svc.ingest_confirm(prev.confirm_token, sid)
    return {
        "status": "success", "report_id": res.report_id, "title": req.title,
        "total_sentences": res.segments_indexed, "extracted_entities": len(prev.entities),
        "extracted_relations": len(prev.relations), "new_nodes_added": res.new_nodes,
        "new_edges_added": res.new_edges, "expires_at": res.expires_at, "duplicate": res.duplicate,
        "warnings": prev.warnings,
    }


@app.get("/api/trace/{verification_id}")
async def trace(verification_id: str, x_session_id: Optional[str] = Header(default=None)):
    return get_service().get_trace(verification_id, x_session_id)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"CyberRAG workbench: http://{host}:{port}")
    uvicorn.run("demo.server:app", host=host, port=port, reload=False)
