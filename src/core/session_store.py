"""Per-session upload indexes and traces with TTL, size limits and locks.

Uploads never touch the shared baseline index (H11): each session owns a small
dense+BM25 index over its own segments. Evidence from it is tagged
``source_type="user_upload"`` and can never count as independent corroboration.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.errors import InvalidInputError, TokenExpiredError

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_TOKEN_RE = re.compile(r"[\w.-]+", re.U)


def validate_session_id(session_id: Optional[str]) -> Optional[str]:
    if session_id is None:
        return None
    if not isinstance(session_id, str) or not SESSION_ID_RE.match(session_id):
        raise InvalidInputError(
            "session_id must be 1-64 characters of letters, digits, '_', '.', ':' or '-'"
        )
    return session_id


def text_fingerprint(text: str) -> str:
    norm = " ".join((text or "").split()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


@dataclass
class PendingIngest:
    token: str
    session_id: str
    report_id: str
    title: str
    publisher: str
    pub_date: str
    fingerprint: str
    segments: List[Dict[str, Any]]
    extraction: Dict[str, Any]
    expires_at: float


@dataclass
class SessionIndex:
    """Small dense + lexical index over one session's uploaded segments."""

    encode: Callable[[List[str]], np.ndarray]
    documents: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    _bm25: Any = None
    _tokens: List[List[str]] = field(default_factory=list)

    def add(self, docs: List[Dict[str, Any]]) -> None:
        if not docs:
            return
        texts = [d["text"] for d in docs]
        emb = np.asarray(self.encode(texts), dtype=np.float32)
        self.embeddings = emb if self.embeddings is None else np.vstack([self.embeddings, emb])
        self.documents.extend(docs)
        self._tokens.extend([_TOKEN_RE.findall(t.lower()) for t in texts])
        self._bm25 = None  # rebuilt lazily

    def _ensure_bm25(self) -> None:
        if self._bm25 is None and self._tokens:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.documents or self.embeddings is None:
            return []
        q = np.asarray(self.encode([query]), dtype=np.float32)[0]
        dense = np.asarray(self.embeddings @ q).reshape(-1)
        self._ensure_bm25()
        bm25 = np.asarray(self._bm25.get_scores(_TOKEN_RE.findall(query.lower()))) if self._bm25 else np.zeros(len(dense))
        fused: Dict[int, float] = {}
        for rank, idx in enumerate(np.argsort(-dense, kind="stable")[: max(top_k, 25)], 1):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(np.argsort(-bm25, kind="stable")[: max(top_k, 25)], 1):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + rank)
        ranked = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        out = []
        for i in ranked:
            doc = dict(self.documents[i])
            doc.update(
                {
                    "rrf_score": float(fused[i]),
                    "dense_score": float(dense[i]),
                    "bm25_score": float(bm25[i]),
                    "retrieval_method": "session",
                    "source_type": "user_upload",
                }
            )
            out.append(doc)
        return out


@dataclass
class Session:
    session_id: str
    created: float
    last_seen: float
    lock: threading.RLock = field(default_factory=threading.RLock)
    index: Optional[SessionIndex] = None
    reports: Dict[str, str] = field(default_factory=dict)  # fingerprint -> report_id
    pending: Dict[str, PendingIngest] = field(default_factory=dict)
    traces: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extra_nodes: List[Dict[str, Any]] = field(default_factory=list)
    extra_edges: List[Dict[str, Any]] = field(default_factory=list)


class SessionStore:
    """Thread-safe store. ``clock`` is injectable for TTL tests."""

    def __init__(
        self,
        ttl_secs: int = 7200,
        max_sessions: int = 256,
        confirm_ttl_secs: int = 600,
        clock: Callable[[], float] = time.time,
    ):
        self.ttl = ttl_secs
        self.max_sessions = max_sessions
        self.confirm_ttl = confirm_ttl_secs
        self._clock = clock
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ core
    def _purge(self) -> None:
        now = self._clock()
        for sid in [s for s, v in self._sessions.items() if now - v.last_seen > self.ttl]:
            del self._sessions[sid]

    def get(self, session_id: Optional[str], create: bool = False) -> Optional[Session]:
        sid = validate_session_id(session_id)
        if sid is None:
            return None
        with self._lock:
            self._purge()
            sess = self._sessions.get(sid)
            if sess is None and create:
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
                    del self._sessions[oldest.session_id]
                now = self._clock()
                sess = Session(session_id=sid, created=now, last_seen=now)
                self._sessions[sid] = sess
            if sess is not None:
                sess.last_seen = self._clock()
            return sess

    def active_count(self) -> int:
        with self._lock:
            self._purge()
            return len(self._sessions)

    def expires_at(self, sess: Session) -> float:
        return sess.last_seen + self.ttl

    # ---------------------------------------------------------------- traces
    def store_trace(self, session_id: Optional[str], verification_id: str, trace: Dict[str, Any]) -> None:
        sess = self.get(session_id, create=session_id is not None)
        with self._lock:
            if sess is None:
                self._global_traces()[verification_id] = trace
                return
            with sess.lock:
                sess.traces[verification_id] = trace

    def _global_traces(self) -> Dict[str, Dict[str, Any]]:
        if not hasattr(self, "_traces"):
            self._traces: Dict[str, Dict[str, Any]] = {}
        # Bounded: keep the newest 2000
        if len(self._traces) > 2000:
            for key in list(self._traces)[: len(self._traces) - 2000]:
                del self._traces[key]
        return self._traces

    def get_trace(self, verification_id: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            tr = self._global_traces().get(verification_id)
            if tr is not None:
                return tr
            sessions = [self.get(session_id)] if session_id else []
        for sess in sessions:
            if sess is not None and verification_id in sess.traces:
                return sess.traces[verification_id]
        return None

    # --------------------------------------------------------------- pending
    def new_pending(self, sess: Session, **kw: Any) -> PendingIngest:
        token = secrets.token_urlsafe(24)
        pend = PendingIngest(
            token=token,
            session_id=sess.session_id,
            expires_at=self._clock() + self.confirm_ttl,
            **kw,
        )
        with sess.lock:
            now = self._clock()
            for t in [t for t, p in sess.pending.items() if p.expires_at < now]:
                del sess.pending[t]
            sess.pending[token] = pend
        return pend

    def take_pending(self, session_id: str, token: str) -> PendingIngest:
        sess = self.get(session_id)
        if sess is None:
            raise TokenExpiredError("Unknown or expired session for this confirm_token")
        with sess.lock:
            pend = sess.pending.get(token)
            if pend is None:
                raise TokenExpiredError("confirm_token is invalid or belongs to another session")
            if pend.expires_at < self._clock():
                del sess.pending[token]
                raise TokenExpiredError("confirm_token expired (10 minute limit)")
            del sess.pending[token]
            return pend
