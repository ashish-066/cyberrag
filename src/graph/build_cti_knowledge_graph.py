"""End-to-End Orchestration Pipeline for CTI Knowledge Graph and Assertion Generation.

Consumes:
- data/processed_segments.jsonl (5,991 segments)
- data/entities.jsonl (8,368 extracted entities)

Produces:
- data/assertions.jsonl (Evidence-backed semantic claims with provenance)
- data/cti_knowledge_graph_trusted.json (Serialized NetworkX MultiDiGraph)
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx

from src.nlp.ner_engine import ExtractedEntity
from src.nlp.relation_extractor import CyberRelationExtractor, ExtractedRelation
from src.preprocessor.segmenter import PreprocessedSentence
from src.graph.builder import CyberKnowledgeGraphBuilder
from src.graph.exporter import CyberKnowledgeGraphExporter

SEGMENTS_FILE = Path("data/processed_segments.jsonl")
ENTITIES_FILE = Path("data/entities.jsonl")
ASSERTIONS_FILE = Path("data/assertions.jsonl")
GRAPH_FILE = "data/cti_knowledge_graph_trusted.json"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON Lines file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def main():
    print("=" * 60)
    print("STARTING CTI KNOWLEDGE GRAPH & ASSERTION PIPELINE")
    print("=" * 60)

    # 1. Load Segments
    print(f"Loading segments from {SEGMENTS_FILE}...")
    segments = read_jsonl(SEGMENTS_FILE)
    segments_by_id = {s["sentence_id"]: s for s in segments}
    print(f"Loaded {len(segments):,} segments.")

    # 2. Load Entities
    print(f"Loading extracted entities from {ENTITIES_FILE}...")
    raw_entities = read_jsonl(ENTITIES_FILE)
    print(f"Loaded {len(raw_entities):,} entities.")

    # Group entities by sentence_id and convert to ExtractedEntity instances
    entities_by_sent: Dict[str, List[ExtractedEntity]] = defaultdict(list)
    entity_mention_lookup: Dict[str, Dict[str, Any]] = {}
    pydantic_entities_all: List[ExtractedEntity] = []

    for ent in raw_entities:
        ent_id = ent["entity_id"]
        entity_mention_lookup[ent_id] = ent
        s_id = ent.get("sentence_id") or ent.get("segment_id", "")
        seg = segments_by_id.get(s_id, {})
        p_id = seg.get("paragraph_id", ent.get("paragraph_id", ""))

        pydantic_ent = ExtractedEntity(
            entity_id=f"{ent['category']}::{ent['canonical_name']}",
            text=ent["entity_text"],
            surface_text=ent["surface_text"],
            category=ent["category"],
            canonical_name=ent["canonical_name"],
            canonical_id=ent.get("canonical_id", ""),
            start_char=ent["start_char"],
            end_char=ent["end_char"],
            confidence=ent.get("confidence", 0.95),
            is_ambiguous=ent.get("is_ambiguous", False),
            sentence_id=s_id,
            paragraph_id=p_id,
            report_id=ent["report_id"],
            source=ent["source"],
            publication_date=ent["publication_date"],
            metadata=ent.get("metadata", {}),
        )
        entities_by_sent[s_id].append(pydantic_ent)
        pydantic_entities_all.append(pydantic_ent)

    # 3. Extract Relationships across Segments using Clause-Aware Extractor
    print("\nExtracting relationships using CyberRelationExtractor...")
    extractor = CyberRelationExtractor()
    extracted_relations: List[ExtractedRelation] = []
    assertions: List[Dict[str, Any]] = []

    for s_idx, seg in enumerate(segments):
        s_id = seg["sentence_id"]
        s_ents = entities_by_sent.get(s_id, [])

        if len(s_ents) < 2:
            # Need at least 2 entities in sentence to form a relationship
            continue

        sentence = PreprocessedSentence(
            sentence_id=seg["sentence_id"],
            paragraph_id=seg["paragraph_id"],
            report_id=seg["report_id"],
            source=seg["source"],
            publication_date=seg["publication_date"],
            paragraph_index=seg["paragraph_index"],
            sentence_index=seg["sentence_index"],
            global_sentence_index=seg["global_sentence_index"],
            text=seg["text"],
            start_char=seg.get("start_char", 0),
            end_char=seg.get("end_char", len(seg["text"])),
        )

        sent_relations = extractor.extract_from_sentence(
            sentence=sentence,
            entities=s_ents,
            context_actor=None,
        )

        for rel in sent_relations:
            extracted_relations.append(rel)

            # Match mention IDs from entities.jsonl
            subj_mention_id = "UNKNOWN"
            obj_mention_id = "UNKNOWN"

            # Find matching entity mentions in this sentence
            for raw_ent in raw_entities:
                if raw_ent["sentence_id"] == s_id:
                    if (
                        raw_ent["canonical_name"] == rel.source_entity
                        and raw_ent["category"] == rel.source_category
                    ):
                        subj_mention_id = raw_ent["entity_id"]
                    if (
                        raw_ent["canonical_name"] == rel.target_entity
                        and raw_ent["category"] == rel.target_category
                    ):
                        obj_mention_id = raw_ent["entity_id"]

            assertion_id = f"A{len(assertions) + 1:06d}"
            assertion_record = {
                "assertion_id": assertion_id,
                "relation_id": rel.relation_id,
                "subject": rel.source_entity,
                "subject_entity_id": subj_mention_id,
                "subject_canonical_id": rel.source_id,
                "subject_id": f"{rel.source_category}::{rel.source_entity}",
                "subject_category": rel.source_category,
                "predicate": rel.relation_type,
                "object": rel.target_entity,
                "object_entity_id": obj_mention_id,
                "object_canonical_id": rel.target_id,
                "object_id": f"{rel.target_category}::{rel.target_entity}",
                "object_category": rel.target_category,
                "claim_text": f"{rel.source_entity} {rel.relation_type} {rel.target_entity}.",
                "evidence_sentence": rel.evidence_text,
                "segment_id": rel.sentence_id,
                "sentence_id": rel.sentence_id,
                "paragraph_id": rel.paragraph_id,
                "report_id": rel.report_id,
                "source": rel.source,
                "source_vendor": rel.source,
                "publication_date": rel.publication_date,
                "confidence": round(rel.confidence, 4),
                "trust_level": "trusted",
                "human_verified": False,
                "polarity": rel.polarity,
                "trigger": rel.trigger,
                "extraction_method": "clause_aware_rule",
                "verification_status": "unverified",
            }
            assertions.append(assertion_record)

    # 4. Save data/assertions.jsonl
    ASSERTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ASSERTIONS_FILE.open("w", encoding="utf-8") as f:
        for a in assertions:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(assertions):,} assertions to {ASSERTIONS_FILE}")

    # 5. Build Knowledge Graph with CyberKnowledgeGraphBuilder
    print("\nBuilding Knowledge Graph using CyberKnowledgeGraphBuilder...")
    builder = CyberKnowledgeGraphBuilder()

    # Add all extracted entities as graph nodes
    builder.add_entities(pydantic_entities_all)

    # Add positive relations as directed typed edges
    positive_relations = [r for r in extracted_relations if r.polarity == "positive"]
    builder.add_relations(positive_relations)
    builder.finalize_nodes()

    # 6. Export Trusted Knowledge Graph
    print(f"Exporting Knowledge Graph to {GRAPH_FILE}...")
    exporter = CyberKnowledgeGraphExporter(graph=builder.graph)
    exporter.export_json(output_path=GRAPH_FILE)

    # 7. Summary Metrics
    predicate_counts = Counter(a["predicate"] for a in assertions)
    polarity_counts = Counter(a["polarity"] for a in assertions)
    total_nodes = builder.graph.number_of_nodes()
    total_edges = builder.graph.number_of_edges()

    # Count isolated nodes (degree == 0)
    isolated_nodes = list(nx.isolates(builder.graph))

    print("\n" + "=" * 60)
    print("ORCHESTRATION PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total assertions generated : {len(assertions):,}")
    print(f"Positive assertions        : {polarity_counts.get('positive', 0):,}")
    print(f"Negative assertions        : {polarity_counts.get('negative', 0):,}")
    print("\nPredicate Distribution:")
    for pred, count in predicate_counts.most_common():
        pct = (count / len(assertions)) * 100 if assertions else 0
        print(f"  {pred:<16}: {count:>5} ({pct:>5.1f}%)")

    print("\nKnowledge Graph Topology:")
    print(f"  Total graph nodes        : {total_nodes:,}")
    print(f"  Total graph edges        : {total_edges:,}")
    print(f"  Connected entities       : {total_nodes - len(isolated_nodes):,}")
    print(f"  Isolated entities        : {len(isolated_nodes):,}")

    print("\n" + "=" * 60)
    print("FIRST 3 ASSERTIONS")
    print("=" * 60)
    for a in assertions[:3]:
        print(json.dumps(a, indent=2, ensure_ascii=False))
        print("-" * 60)


if __name__ == "__main__":
    main()
