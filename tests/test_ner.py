"""Tests for Cybersecurity NER and Entity Normalization."""

from src.nlp.normalizer import EntityNormalizer
from src.nlp.ner_engine import CyberNEREngine
from src.preprocessor.segmenter import PreprocessedSentence


def test_entity_normalizer():
    normalizer = EntityNormalizer()

    # Threat Actor alias normalization
    res1 = normalizer.normalize("Vanguard Panda")
    assert res1["canonical_name"] == "Volt Typhoon"
    assert res1["category"] == "ThreatActor"
    assert res1["canonical_id"] == "TA-CN-0001"

    res2 = normalizer.normalize("Midnight Blizzard")
    assert res2["canonical_name"] == "APT29"
    assert res2["category"] == "ThreatActor"

    # Named CVE alias normalization
    res3 = normalizer.normalize("Citrix Bleed")
    assert res3["canonical_name"] == "CVE-2023-4966"
    assert res3["category"] == "CVE"

    # Standard CVE format normalization
    res4 = normalizer.normalize("cve-2021-44228")
    assert res4["canonical_name"] == "CVE-2021-44228"
    assert res4["category"] == "CVE"

    # Technique normalization
    res5 = normalizer.normalize("T1566")
    assert "Phishing" in res5["canonical_name"]
    assert res5["category"] == "Technique"


def test_ner_engine_extraction():
    engine = CyberNEREngine()

    sentence = PreprocessedSentence(
        sentence_id="RPT::p0::s0",
        paragraph_id="RPT::p0",
        report_id="RPT",
        source="CISA",
        publication_date="2024-01-01",
        paragraph_index=0,
        sentence_index=0,
        global_sentence_index=0,
        text="Fancy Bear exploited CVE-2023-38831 using Cobalt Strike against Microsoft in Ukraine.",
    )

    entities = engine.extract_from_sentence(sentence)
    ent_map = {e.category: [] for e in entities}
    for e in entities:
        ent_map[e.category].append(e.canonical_name)

    # Check Threat Actor
    assert "ThreatActor" in ent_map
    assert "APT28" in ent_map["ThreatActor"]

    # Check CVE
    assert "CVE" in ent_map
    assert "CVE-2023-38831" in ent_map["CVE"]

    # Check Tool / Framework
    assert "Tool" in ent_map
    assert "Cobalt Strike" in ent_map["Tool"]

    # Check Organization
    assert "Organization" in ent_map
    assert "Microsoft" in ent_map["Organization"]

    # Check Location
    assert "Location" in ent_map
    assert "Ukraine" in ent_map["Location"]

    # Verify provenance on entity
    first_ent = entities[0]
    assert first_ent.report_id == "RPT"
    assert first_ent.source == "CISA"
    assert first_ent.publication_date == "2024-01-01"
