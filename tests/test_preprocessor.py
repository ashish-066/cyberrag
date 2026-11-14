"""Tests for CTI cleaning and segmentation preserving provenance."""

from src.dataset.loader import CTIReport
from src.preprocessor.cleaner import CTICleaner
from src.preprocessor.segmenter import CTISegmenter


def test_cleaner_defanging():
    cleaner = CTICleaner()
    obfuscated = "The malware connects to hxxps[:]//evil[.]domain[.]com:8080/c2 and sends alerts to admin[@]corp[.]net."
    cleaned = cleaner.clean(obfuscated)
    assert "https://evil.domain.com:8080/c2" in cleaned
    assert "admin@corp.net" in cleaned
    assert "hxxps" not in cleaned
    assert "[.]" not in cleaned


def test_cleaner_unicode_normalization():
    cleaner = CTICleaner()
    text = "“Threat actors—including APT28—used ‘living-off-the-land’ techniques.”"
    cleaned = cleaner.clean(text)
    assert '"Threat actors-including APT28-used \'living-off-the-land\' techniques."' in cleaned


def test_segmenter_provenance_preservation():
    cleaner = CTICleaner()
    segmenter = CTISegmenter(cleaner)

    raw_report = CTIReport(
        report_id="TEST-REPORT-001",
        source="CISA",
        publication_date="2024-03-01",
        title="Sample Threat Advisory",
        content=(
            "Volt Typhoon targeted critical infrastructure across the U.S. and Guam. "
            "The actor exploited CVE-2024-21887 in Ivanti appliances.\n\n"
            "Once inside, the operators used PowerShell and Mimikatz. "
            "They utilized T1190 Exploit Public-Facing Application."
        ),
    )

    preprocessed = segmenter.process_report(raw_report)

    assert preprocessed.report_id == "TEST-REPORT-001"
    assert preprocessed.source == "CISA"
    assert preprocessed.publication_date == "2024-03-01"
    assert preprocessed.total_paragraphs == 2
    assert preprocessed.total_sentences == 4

    # Verify first paragraph & sentences
    p0 = preprocessed.paragraphs[0]
    assert p0.paragraph_id == "TEST-REPORT-001::p0"
    assert p0.report_id == "TEST-REPORT-001"
    assert p0.source == "CISA"
    assert p0.publication_date == "2024-03-01"
    assert len(p0.sentences) == 2

    s0 = p0.sentences[0]
    assert s0.sentence_id == "TEST-REPORT-001::p0::s0"
    assert s0.report_id == "TEST-REPORT-001"
    assert s0.source == "CISA"
    assert s0.publication_date == "2024-03-01"
    assert "Volt Typhoon targeted critical infrastructure across the U.S. and Guam." in s0.text

    s1 = p0.sentences[1]
    assert s1.sentence_id == "TEST-REPORT-001::p0::s1"
    assert "CVE-2024-21887" in s1.text
