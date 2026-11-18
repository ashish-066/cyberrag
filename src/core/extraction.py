"""Report extraction for uploads: segmentation, NER and relation extraction.

Used by the two-step ingest (preview -> confirm). Nothing here mutates the
shared corpus. Live NER is the gazetteer engine only: v5 built an *untrained*
ML model that silently returned nothing, so ML NER is disabled and the preview
says so (H15).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.preprocessor.cleaner import CTICleaner
from src.preprocessor.segmenter import CTISegmenter, PreprocessedSentence

ML_NER_NOTE = "ML NER is not active (no trained model artifact is shipped); gazetteer NER only."


class ReportExtractor:
    def __init__(
        self,
        known_node_ids: Optional[Iterable[str]] = None,
        known_edges: Optional[Iterable[Tuple[str, str, str]]] = None,
        ner_engine: Any = None,
        relation_extractor: Any = None,
    ):
        self.cleaner = CTICleaner()
        self.segmenter = CTISegmenter(cleaner=self.cleaner)
        self._ner = ner_engine
        self._rel = relation_extractor
        self.known_node_ids: Set[str] = set(known_node_ids or [])
        self.known_edges: Set[Tuple[str, str, str]] = set(known_edges or [])
        self.ml_ner_active = False

    @property
    def ner(self):
        if self._ner is None:
            from src.nlp.ner_engine import CyberNEREngine

            self._ner = CyberNEREngine(enable_ml=False)
        return self._ner

    @property
    def relations(self):
        if self._rel is None:
            from src.nlp.relation_extractor import CyberRelationExtractor

            self._rel = CyberRelationExtractor()
        return self._rel

    def sentences(self, text: str, report_id: str, publisher: str, pub_date: str) -> List[PreprocessedSentence]:
        clean = self.cleaner.clean(text)
        out: List[PreprocessedSentence] = []
        g = 0
        for p_idx, para in enumerate(clean.split("\n\n")):
            para = para.strip()
            if not para:
                continue
            p_id = f"{report_id}::p{p_idx}"
            for s_idx, (s_text, s_start, s_end) in enumerate(self.segmenter.segment_sentences(para)):
                if not s_text.strip():
                    continue
                out.append(
                    PreprocessedSentence(
                        sentence_id=f"{p_id}::s{s_idx}", paragraph_id=p_id, report_id=report_id,
                        source=publisher, publication_date=pub_date, paragraph_index=p_idx,
                        sentence_index=s_idx, global_sentence_index=g, text=s_text,
                        start_char=s_start, end_char=s_end,
                    )
                )
                g += 1
        return out

    def extract(self, text: str, *, report_id: str, publisher: str, pub_date: str, title: str = "") -> Dict[str, Any]:
        sents = self.sentences(text, report_id, publisher, pub_date)
        entities_by_sent: Dict[str, list] = {}
        ents: Dict[str, Dict[str, Any]] = {}
        for s in sents:
            found = self.ner.extract_from_sentence(s)
            entities_by_sent[s.sentence_id] = found
            for e in found:
                rec = ents.setdefault(
                    e.entity_id, {"id": e.entity_id, "name": e.canonical_name, "category": e.category, "mentions": 0}
                )
                rec["mentions"] += 1
        rels: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_edges: Set[Tuple[str, str, str]] = set()
        for s in sents:
            found = entities_by_sent.get(s.sentence_id, [])
            if len(found) < 2:
                continue
            for r in self.relations.extract_from_sentence(s, found):
                src_id = f"{r.source_category}::{r.source_entity}"
                tgt_id = f"{r.target_category}::{r.target_entity}"
                rels.append(
                    {
                        "subject": r.source_entity, "relation": r.relation_type, "object": r.target_entity,
                        "polarity": getattr(r, "polarity", "positive"), "trigger": r.trigger,
                        "confidence": r.confidence, "evidence_text": r.evidence_text, "sentence_id": r.sentence_id,
                    }
                )
                key = (src_id, tgt_id, r.relation_type)
                if getattr(r, "polarity", "positive") == "positive" and key not in self.known_edges and key not in seen_edges:
                    seen_edges.add(key)
                    edges.append({"source": src_id, "target": tgt_id, "relation_type": r.relation_type, "report_id": report_id})
        segments = [
            {
                "report_id": report_id, "segment_id": s.sentence_id, "sentence_id": s.sentence_id,
                "text": s.text, "clean_text": s.text, "source": publisher, "publication_date": pub_date,
                "title": title, "source_type": "user_upload",
            }
            for s in sents
        ]
        nodes = [
            {"id": eid, "name": rec["name"], "category": rec["category"], "report_id": report_id}
            for eid, rec in ents.items()
            if eid not in self.known_node_ids
        ]
        return {
            "segments": segments,
            "entities": sorted(ents.values(), key=lambda r: (-r["mentions"], r["id"])),
            "relations": rels,
            "graph_nodes": nodes,
            "graph_edges": edges,
            "ml_ner_active": self.ml_ner_active,
            "notes": [ML_NER_NOTE],
        }
