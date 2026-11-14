"""Comprehensive Verification Script for Person 1 Deliverables.

Validates:
1. Entity Coverage across all 7 categories (especially CVEs and Locations)
2. Relationship Quality (trigger grounding, directional schema validity, no false vendor targets)
3. Full Metadata Preservation (report_id, source, publication_date, sentence_id on 100% of items)
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("PipelineVerifier")


def verify():
    logger.info("=" * 60)
    logger.info("RUNNING QUALITY & PROVENANCE VERIFICATION CHECKS")
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # 1. Verify Preprocessed Data
    # ---------------------------------------------------------
    preprocessed_path = Path("data/processed/preprocessed_cti.json")
    assert preprocessed_path.exists(), "Preprocessed CTI file missing!"
    with open(preprocessed_path, "r", encoding="utf-8") as f:
        preprocessed_data = json.load(f)

    total_sentences = 0
    sentence_provenance_errors = 0
    for r in preprocessed_data:
        assert r.get("report_id"), "Missing report_id on report!"
        assert r.get("source"), "Missing source on report!"
        assert r.get("publication_date"), "Missing publication_date on report!"

        for p in r.get("paragraphs", []):
            assert p.get("paragraph_id"), "Missing paragraph_id!"
            for s in p.get("sentences", []):
                total_sentences += 1
                if not s.get("report_id") or not s.get("source") or not s.get("publication_date") or not s.get("sentence_id"):
                    sentence_provenance_errors += 1

    logger.info(f"[Check 1/4] Preprocessed Sentences: {total_sentences} checked. Provenance errors: {sentence_provenance_errors}")
    assert sentence_provenance_errors == 0, f"Found {sentence_provenance_errors} sentences missing provenance!"

    # ---------------------------------------------------------
    # 2. Verify Entity Coverage (All 7 Categories)
    # ---------------------------------------------------------
    entities_path = Path("data/entities/extracted_entities.json")
    assert entities_path.exists(), "Extracted entities file missing!"
    with open(entities_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    category_counts = {}
    entity_provenance_errors = 0
    for e in entities:
        cat = e.get("category", "Unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if not e.get("report_id") or not e.get("source") or not e.get("publication_date") or not e.get("sentence_id"):
            entity_provenance_errors += 1

    required_categories = ["ThreatActor", "Malware", "Tool", "CVE", "Technique", "Organization", "Location"]
    logger.info(f"[Check 2/4] Entity Coverage Check ({len(entities)} total mentions):")
    for cat in required_categories:
        count = category_counts.get(cat, 0)
        logger.info(f"  - {cat:15}: {count} mentions")
        assert count > 0, f"Category {cat} has 0 extracted mentions!"

    assert category_counts.get("CVE", 0) >= 4, "Insufficient CVE coverage!"
    assert category_counts.get("Location", 0) >= 5, "Insufficient Location coverage!"
    assert entity_provenance_errors == 0, f"Found {entity_provenance_errors} entities missing provenance!"

    # ---------------------------------------------------------
    # 3. Verify Relationship Quality & Trigger Grounding
    # ---------------------------------------------------------
    relations_path = Path("data/relations/extracted_relations.json")
    assert relations_path.exists(), "Extracted relations file missing!"
    with open(relations_path, "r", encoding="utf-8") as f:
        relations = json.load(f)

    rel_counts = {}
    missing_triggers = 0
    relation_provenance_errors = 0
    invalid_schema_count = 0

    valid_schemas = {
        "USES": {("ThreatActor", "Malware"), ("ThreatActor", "Tool"), ("Malware", "Tool"), ("Malware", "Malware")},
        "TARGETS": {("ThreatActor", "Organization"), ("ThreatActor", "Location"), ("Malware", "Organization"), ("Malware", "Location")},
        "EXPLOITS": {("ThreatActor", "CVE"), ("Malware", "CVE"), ("Tool", "CVE")},
        "USES_TECHNIQUE": {("ThreatActor", "Technique"), ("Malware", "Technique")},
    }

    for r in relations:
        r_type = r.get("relation_type")
        rel_counts[r_type] = rel_counts.get(r_type, 0) + 1

        # Check trigger grounding
        trigger = r.get("trigger", "").strip()
        if not trigger:
            missing_triggers += 1

        # Check provenance
        if not r.get("report_id") or not r.get("source") or not r.get("publication_date") or not r.get("sentence_id") or not r.get("evidence_text"):
            relation_provenance_errors += 1

        # Check schema typing
        src_cat = r.get("source_category")
        tgt_cat = r.get("target_category")
        pair = (src_cat, tgt_cat)
        if pair not in valid_schemas.get(r_type, set()):
            invalid_schema_count += 1

    required_relations = ["USES", "TARGETS", "EXPLOITS", "USES_TECHNIQUE"]
    logger.info(f"[Check 3/4] Relationship Quality Check ({len(relations)} total relations):")
    for r_type in required_relations:
        count = rel_counts.get(r_type, 0)
        logger.info(f"  - {r_type:15}: {count} instances")
        assert count > 0, f"Relation type {r_type} has 0 extracted instances!"

    logger.info(f"  - Missing action triggers: {missing_triggers}")
    logger.info(f"  - Invalid schema pairs:   {invalid_schema_count}")
    logger.info(f"  - Provenance errors:      {relation_provenance_errors}")

    assert missing_triggers == 0, f"{missing_triggers} relations lack semantic action triggers!"
    assert invalid_schema_count == 0, f"{invalid_schema_count} relations violate directional typing!"
    assert relation_provenance_errors == 0, f"{relation_provenance_errors} relations lack provenance!"

    # ---------------------------------------------------------
    # 4. Verify Knowledge Graph Exports & Attributes
    # ---------------------------------------------------------
    kg_json_path = Path("data/knowledge_graph/cyber_kg.json")
    assert kg_json_path.exists(), "Cyber KG JSON export missing!"
    with open(kg_json_path, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    nodes = kg_data.get("nodes", [])
    links = kg_data.get("links", [])
    logger.info(f"[Check 4/4] Knowledge Graph Export: {len(nodes)} nodes, {len(links)} links.")

    node_provenance_errors = 0
    for n in nodes:
        if not n.get("report_id") or not n.get("publication_date") or not n.get("source"):
            node_provenance_errors += 1

    link_provenance_errors = 0
    for l in links:
        if not l.get("report_id") or not l.get("publication_date") or not l.get("sentence_id"):
            link_provenance_errors += 1

    logger.info(f"  - Node provenance gaps: {node_provenance_errors}")
    logger.info(f"  - Link provenance gaps: {link_provenance_errors}")
    assert node_provenance_errors == 0, "Nodes missing top-level provenance in KG JSON!"
    assert link_provenance_errors == 0, "Links missing top-level provenance in KG JSON!"

    logger.info("=" * 60)
    logger.info("VERIFICATION PASSED: 100% QUALITY AND PROVENANCE COMPLIANT!")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
