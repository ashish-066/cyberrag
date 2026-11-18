"""
Decision Engine and Entity-Aware Evidence Gating for CyberRAG.
Prevents false NLI contradictions by strictly filtering candidate passages for endpoint entity presence.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from src.llm.query_expander import QueryExpander, KNOWN_ALIASES


class EntityAwareEvidenceGate:
    """
    Prevents false NLI contradictions by strictly filtering candidate passages
    for endpoint entity presence (including known aliases).
    """

    PARTIAL_OVERLAP_DISTINCT_ENTITIES: Dict[str, List[str]] = {
        "lazarus": [r"\bfancy\s+lazarus\b"],
        "lazarus group": [r"\bfancy\s+lazarus\b"],
        "bear": [r"\b(?:fancy|cozy|energetic|venomous|voodoo)\s+bear\b"],
    }

    def __init__(self, expander: Optional[QueryExpander] = None):
        self.expander = expander or QueryExpander()

    def get_search_terms(self, entity: str) -> Set[str]:
        return self.expander.get_search_terms(entity)

    def _entity_present(self, text_lower: str, terms: Set[str], canonical: str) -> bool:
        clean = text_lower
        for pattern in self.PARTIAL_OVERLAP_DISTINCT_ENTITIES.get(canonical.lower(), []):
            clean = re.sub(pattern, " ", clean)
        return any(re.search(rf"\b{re.escape(t)}\b", clean) for t in terms)

    def is_bibliography_or_footnote(self, text: str) -> bool:
        """
        Detect citation blocks, footnote back-references, URL link lists, and bibliography headers.
        Clean narrative sentences containing inline citation markers (e.g. '...APT28 [[1]] (# _ ftn1)')
        are NOT rejected; genuine bibliography sections have back-references ('ftnref'),
        dense external URLs (>= 2), or bibliographic list headers.
        """
        t_lower = (text or "").lower()
        if re.search(r"(ftnref|references:\s*\[|sources:\s*\[|bibliography:\s*\[)", t_lower):
            return True
        url_count = len(re.findall(r"https?:\/\/", t_lower))
        link_count = len(re.findall(r"\[.*?\]\s*\(https?:\/\/.*?\)", t_lower))
        if url_count >= 2 or link_count >= 2:
            return True
        if re.search(r"[-_*]{3,}\s*\[\[?\d+\]?\]", t_lower):
            return True
        return False

    def is_probative(self, passage_text: str, subject: str, obj: str) -> Tuple[bool, str]:
        # 1. Reject bibliography, citation references, and footnote metadata blocks
        if self.is_bibliography_or_footnote(passage_text):
            return False, "BIBLIOGRAPHY_OR_FOOTNOTE_REJECTED"

        text_l = (passage_text or "").lower()
        s_terms = self.get_search_terms(subject)
        o_terms = self.get_search_terms(obj)

        has_s_passage = self._entity_present(text_l, s_terms, subject)
        has_o_passage = self._entity_present(text_l, o_terms, obj)

        if not has_s_passage and not has_o_passage:
            return False, "NEITHER_ENTITY_PRESENT"
        elif not has_s_passage:
            return False, f"MISSING_SUBJECT_{subject.upper()}"
        elif not has_o_passage:
            return False, f"MISSING_OBJECT_{obj.upper()}"

        # 2. Strict Sentence-Level Co-occurrence: Subject and Object must co-occur
        # within the SAME sentence/clause, not merely within the same 1,000-character paragraph.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", passage_text or "") if s.strip()]
        if not sentences:
            sentences = [passage_text.strip()]

        for sent in sentences:
            sent_l = sent.lower()
            sent_has_s = self._entity_present(sent_l, s_terms, subject)
            sent_has_o = self._entity_present(sent_l, o_terms, obj)
            if sent_has_s and sent_has_o:
                return True, "PROBATIVE_COOCCURRENCE"

        return False, "CROSS_SENTENCE_COOCCURRENCE_REJECTED"

    def filter_evidence_list(
        self, items: Sequence[Dict[str, Any]], subject: str, obj: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        probative = []
        rejected = []
        for item in items:
            text = item.get("text") or item.get("evidence") or ""
            ok, reason = self.is_probative(text, subject, obj)
            item_copy = dict(item)
            item_copy["gate_status"] = reason
            if ok:
                probative.append(item_copy)
            else:
                rejected.append(item_copy)
        return probative, rejected
