"""Cyber Knowledge Graph Builder using NetworkX.

Constructs a multi-relational, typed cybersecurity knowledge graph from extracted
entities and relationships with full factual provenance.
"""

import logging
from typing import List, Dict, Any, Optional
import networkx as nx

from src.nlp.ner_engine import ExtractedEntity
from src.nlp.relation_extractor import ExtractedRelation

logger = logging.getLogger(__name__)


class CyberKnowledgeGraphBuilder:
    """Builds and enriches a NetworkX MultiDiGraph representing the Cyber Knowledge Graph."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_entities(self, entities: List[ExtractedEntity]) -> None:
        """Add and aggregate entity mentions into graph nodes."""
        for ent in entities:
            node_id = f"{ent.category}::{ent.canonical_name}"

            if not self.graph.has_node(node_id):
                self.graph.add_node(
                    node_id,
                    name=ent.canonical_name,
                    category=ent.category,
                    canonical_id=ent.canonical_id,
                    mention_count=1,
                    # Provenance tracking (both primary scalar and collections)
                    report_id=ent.report_id,
                    source=ent.source,
                    publication_date=ent.publication_date,
                    sentence_id=ent.sentence_id,
                    sources={ent.source},
                    reports={ent.report_id},
                    dates={ent.publication_date},
                    surface_forms={ent.text},
                    metadata=ent.metadata or {},
                )
            else:
                node = self.graph.nodes[node_id]
                node["mention_count"] += 1
                node["sources"].add(ent.source)
                node["reports"].add(ent.report_id)
                node["dates"].add(ent.publication_date)
                node["surface_forms"].add(ent.text)

        logger.info(f"Populated {self.graph.number_of_nodes()} unique nodes in the Cyber KG.")

    def add_relations(self, relations: List[ExtractedRelation]) -> None:
        """Add and aggregate relationships into graph edges with full provenance evidence."""
        for rel in relations:
            u = f"{rel.source_category}::{rel.source_entity}"
            v = f"{rel.target_category}::{rel.target_entity}"

            # Ensure nodes exist
            if not self.graph.has_node(u):
                self.graph.add_node(
                    u,
                    name=rel.source_entity,
                    category=rel.source_category,
                    canonical_id=rel.source_id,
                    mention_count=1,
                    report_id=rel.report_id,
                    source=rel.source,
                    publication_date=rel.publication_date,
                    sentence_id=rel.sentence_id,
                    sources={rel.source},
                    reports={rel.report_id},
                    dates={rel.publication_date},
                    surface_forms={rel.source_entity},
                    metadata={},
                )
            if not self.graph.has_node(v):
                self.graph.add_node(
                    v,
                    name=rel.target_entity,
                    category=rel.target_category,
                    canonical_id=rel.target_id,
                    mention_count=1,
                    report_id=rel.report_id,
                    source=rel.source,
                    publication_date=rel.publication_date,
                    sentence_id=rel.sentence_id,
                    sources={rel.source},
                    reports={rel.report_id},
                    dates={rel.publication_date},
                    surface_forms={rel.target_entity},
                    metadata={},
                )

            evidence_item = {
                "report_id": rel.report_id,
                "source": rel.source,
                "publication_date": rel.publication_date,
                "sentence_id": rel.sentence_id,
                "paragraph_id": rel.paragraph_id,
                "evidence_text": rel.evidence_text,
                "trigger": rel.trigger,
                "confidence": rel.confidence,
                "polarity": "positive",
                "observation_time": rel.publication_date,
                "scope": "sentence_level",
            }

            key = rel.relation_type
            conf_level = "HIGH" if rel.confidence >= 0.85 else "MEDIUM"
            if self.graph.has_edge(u, v, key=key):
                edge = self.graph[u][v][key]
                edge["evidence_count"] += 1
                edge["evidence"].append(evidence_item)
                edge["confidence"] = max(edge["confidence"], rel.confidence)
                # Ensure sets for aggregation
                if "reports" in edge and isinstance(edge["reports"], set):
                    edge["reports"].add(rel.report_id)
                if "sources" in edge and isinstance(edge["sources"], set):
                    edge["sources"].add(rel.source)
                if "dates" in edge and isinstance(edge["dates"], set):
                    edge["dates"].add(rel.publication_date)
            else:
                self.graph.add_edge(
                    u,
                    v,
                    key=key,
                    relation_type=rel.relation_type,
                    confidence=rel.confidence,
                    confidence_level=conf_level,
                    polarity="positive",
                    observation_time=rel.publication_date,
                    scope="sentence_level",
                    evidence_count=1,
                    # Top-level provenance fields
                    report_id=rel.report_id,
                    source=rel.source,
                    publication_date=rel.publication_date,
                    sentence_id=rel.sentence_id,
                    evidence_text=rel.evidence_text,
                    trigger=rel.trigger,
                    reports={rel.report_id},
                    sources={rel.source},
                    dates={rel.publication_date},
                    evidence=[evidence_item],
                )

        logger.info(f"Populated {self.graph.number_of_edges()} typed edges in the Cyber KG.")

    def finalize_nodes(self) -> None:
        """Convert sets to sorted lists and compute first_seen / last_seen dates."""
        for _, data in self.graph.nodes(data=True):
            if "sources" in data and isinstance(data["sources"], set):
                data["sources"] = sorted(list(data["sources"]))
            if "reports" in data and isinstance(data["reports"], set):
                data["reports"] = sorted(list(data["reports"]))
                if "report_id" not in data or not data["report_id"]:
                    data["report_id"] = data["reports"][0] if data["reports"] else "UNKNOWN"
            if "surface_forms" in data and isinstance(data["surface_forms"], set):
                data["surface_forms"] = sorted(list(data["surface_forms"]))
            if "dates" in data and isinstance(data["dates"], set):
                dates = sorted([d for d in data["dates"] if d and d != "UNKNOWN"])
                data["dates"] = dates
                data["first_seen"] = dates[0] if dates else "UNKNOWN"
                data["last_seen"] = dates[-1] if dates else "UNKNOWN"
                if "publication_date" not in data or not data["publication_date"]:
                    data["publication_date"] = data["first_seen"]

        for _, _, _, data in self.graph.edges(keys=True, data=True):
            if "reports" in data and isinstance(data["reports"], set):
                data["reports"] = sorted(list(data["reports"]))
            if "sources" in data and isinstance(data["sources"], set):
                data["sources"] = sorted(list(data["sources"]))
            if "dates" in data and isinstance(data["dates"], set):
                data["dates"] = sorted(list(data["dates"]))

    def build(
        self,
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
    ) -> nx.MultiDiGraph:
        """Build and finalize complete Knowledge Graph."""
        self.add_entities(entities)
        self.add_relations(relations)
        self.finalize_nodes()
        return self.graph
