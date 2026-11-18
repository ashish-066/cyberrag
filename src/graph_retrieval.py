"""Evidence-preserving retrieval over the trusted CTI graph assertions."""
from __future__ import annotations
import json
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def _whole_phrase_present(phrase: str, text: str) -> bool:
    phrase = phrase.strip()
    if not phrase:
        return False
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I) is not None


class GraphRAGRetriever:
    """
    Matches entity names in a query and retrieves assertions from the graph.
    If two or more query entities are matched, only assertions directly linking
    the matched entity IDs are returned. If one entity is matched, its adjacent
    assertions are returned. No assertion is promoted to a verified fact.
    """

    def __init__(
        self,
        graph_path: str | Path = "data/cti_knowledge_graph_trusted.json",
        assertions_path: str | Path = "data/assertions.jsonl",
        graph: dict[str, Any] | None = None,
        assertions: list[dict[str, Any]] | None = None,
    ):
        if graph is None:
            with open(graph_path, "r", encoding="utf-8") as f:
                graph = json.load(f)
        self.graph = graph
        self.assertions = assertions if assertions is not None else _load_jsonl(assertions_path)

        self.entity_lookup: dict[str, str] = {}
        # Sort nodes by mention_count ascending so dominant nodes take precedence in lowercase collision
        nodes_sorted = sorted(
            self.graph.get("nodes", []),
            key=lambda x: (x.get("mention_count", 0), len(str(x.get("name") or "")))
        )
        for node in nodes_sorted:
            if (node.get("node_type") == "ENTITY" or node.get("category") or "node_type" not in node) and node.get("id"):
                name = str(node.get("name") or "").strip()
                if name:
                    self.entity_lookup[name.lower()] = node["id"]
                for sf in node.get("surface_forms", []):
                    sf_str = str(sf).strip()
                    if sf_str:
                        self.entity_lookup[sf_str.lower()] = node["id"]

        self.entity_index = defaultdict(list)
        for assertion in self.assertions:
            for field in ("subject_id", "object_id"):
                entity_id = assertion.get(field)
                if entity_id:
                    self.entity_index[entity_id].append(assertion)
                    self.entity_index[entity_id.lower()].append(assertion)

        logger.info("Graph Retrieval initialized!")
        logger.info("Entities indexed: %d", len(self.entity_lookup))
        logger.info("Assertions loaded: %d", len(self.assertions))

    def match_entities(self, query: str) -> list[tuple[str, str]]:
        q = query.lower()
        matches = []
        for name, entity_id in self.entity_lookup.items():
            if _whole_phrase_present(name, q):
                matches.append((name, entity_id))
        # Prefer longest names where one name is a substring of another.
        matches.sort(key=lambda x: len(x[0]), reverse=True)
        chosen = []
        for name, entity_id in matches:
            if not any(_whole_phrase_present(name, existing) for existing, _ in chosen):
                chosen.append((name, entity_id))
        return chosen

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        matched = self.match_entities(query)
        entity_ids = {entity_id for _, entity_id in matched}
        if not entity_ids:
            return []

        candidates: dict[str, dict[str, Any]] = {}
        for entity_id in entity_ids:
            for assertion in self.entity_index.get(entity_id, []) + self.entity_index.get(entity_id.lower(), []):
                aid = assertion.get("assertion_id")
                if aid is not None:
                    candidates[str(aid)] = assertion.copy()

        results = list(candidates.values())
        if len(entity_ids) >= 2:
            # Require the assertion endpoints to be the matched query entities.
            entity_ids_lower = {eid.lower() for eid in entity_ids}
            results = [
                item for item in results
                if (item.get("subject_id") in entity_ids or (item.get("subject_id") or "").lower() in entity_ids_lower)
                and (item.get("object_id") in entity_ids or (item.get("object_id") or "").lower() in entity_ids_lower)
            ]

        trust_priority = {"trusted": 3, "candidate": 2, "low": 1}
        confidence_priority = {"high": 3, "medium": 2, "low": 1}
        results.sort(
            key=lambda x: (
                trust_priority.get(x.get("trust_level"), 0),
                confidence_priority.get(x.get("confidence"), 0),
                str(x.get("publication_date") or ""),
            ),
            reverse=True,
        )
        for item in results:
            item["retrieval_method"] = "graph"
            item["matched_entities"] = [name for name, _ in matched]
            item["human_verified"] = bool(item.get("human_verified", False))
        return results[:top_k]
