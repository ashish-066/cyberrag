"""Tests for Relationship Extraction with factual evidence provenance."""

from src.preprocessor.segmenter import PreprocessedSentence
from src.nlp.ner_engine import CyberNEREngine
from src.nlp.relation_extractor import CyberRelationExtractor


def test_relation_extraction_uses():
    ner_engine = CyberNEREngine()
    extractor = CyberRelationExtractor(ner_engine=ner_engine)

    sent = PreprocessedSentence(
        sentence_id="RPT1::p0::s0",
        paragraph_id="RPT1::p0",
        report_id="RPT1",
        source="Mandiant",
        publication_date="2024-02-15",
        paragraph_index=0,
        sentence_index=0,
        global_sentence_index=0,
        text="APT29 deployed Cobalt Strike and Mimikatz for lateral movement.",
    )

    entities = ner_engine.extract_from_sentence(sent)
    relations = extractor.extract_from_sentence(sent, entities)

    rel_types = [r.relation_type for r in relations]
    assert "USES" in rel_types

    uses_rels = [r for r in relations if r.relation_type == "USES"]
    targets = [r.target_entity for r in uses_rels]
    assert "Cobalt Strike" in targets
    assert "Mimikatz" in targets

    # Provenance verification
    first_rel = uses_rels[0]
    assert first_rel.report_id == "RPT1"
    assert first_rel.source == "Mandiant"
    assert first_rel.publication_date == "2024-02-15"
    assert first_rel.evidence_text == sent.text


def test_relation_extraction_exploits():
    ner_engine = CyberNEREngine()
    extractor = CyberRelationExtractor(ner_engine=ner_engine)

    sent = PreprocessedSentence(
        sentence_id="RPT2::p0::s0",
        paragraph_id="RPT2::p0",
        report_id="RPT2",
        source="CISA",
        publication_date="2023-12-10",
        paragraph_index=0,
        sentence_index=0,
        global_sentence_index=0,
        text="Volt Typhoon exploited CVE-2024-21887 in perimeter gateways.",
    )

    entities = ner_engine.extract_from_sentence(sent)
    relations = extractor.extract_from_sentence(sent, entities)

    exploits_rels = [r for r in relations if r.relation_type == "EXPLOITS"]
    assert len(exploits_rels) >= 1
    assert exploits_rels[0].source_entity == "Volt Typhoon"
    assert exploits_rels[0].target_entity == "CVE-2024-21887"
    assert exploits_rels[0].confidence >= 0.85


def test_relation_extraction_targets():
    ner_engine = CyberNEREngine()
    extractor = CyberRelationExtractor(ner_engine=ner_engine)

    sent = PreprocessedSentence(
        sentence_id="RPT3::p0::s0",
        paragraph_id="RPT3::p0",
        report_id="RPT3",
        source="CrowdStrike",
        publication_date="2023-11-01",
        paragraph_index=0,
        sentence_index=0,
        global_sentence_index=0,
        text="Sandworm targeted the Ukrainian Ministry of Defense and electric substations across Ukraine.",
    )

    entities = ner_engine.extract_from_sentence(sent)
    relations = extractor.extract_from_sentence(sent, entities)

    targets_rels = [r for r in relations if r.relation_type == "TARGETS"]
    assert len(targets_rels) >= 1
    target_names = [r.target_entity for r in targets_rels]
    assert "Ukrainian Ministry of Defense" in target_names or "Ukraine" in target_names
