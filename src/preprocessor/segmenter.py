"""Paragraph and sentence segmenter preserving CTI provenance metadata."""

import re
from typing import List
from pydantic import BaseModel, Field

from src.dataset.loader import CTIReport
from .cleaner import CTICleaner


class PreprocessedSentence(BaseModel):
    """A segmented sentence preserving provenance down to sentence level."""

    sentence_id: str = Field(..., description="Unique sentence ID (report_id::pX::sY)")
    paragraph_id: str = Field(..., description="Parent paragraph ID (report_id::pX)")
    report_id: str = Field(..., description="Parent CTI report ID")
    source: str = Field(..., description="Intel source (e.g., CISA, Mandiant)")
    publication_date: str = Field(..., description="Publication date of the report")
    paragraph_index: int = Field(..., description="Zero-based paragraph index")
    sentence_index: int = Field(..., description="Zero-based sentence index within paragraph")
    global_sentence_index: int = Field(..., description="Zero-based sentence index within report")
    text: str = Field(..., description="Cleaned sentence text")
    start_char: int = Field(default=0, description="Start char offset in paragraph")
    end_char: int = Field(default=0, description="End char offset in paragraph")


class PreprocessedParagraph(BaseModel):
    """A segmented paragraph containing structured sentences."""

    paragraph_id: str = Field(..., description="Unique paragraph ID (report_id::pX)")
    report_id: str = Field(..., description="Parent CTI report ID")
    paragraph_index: int = Field(..., description="Zero-based paragraph index")
    source: str = Field(..., description="Intel source")
    publication_date: str = Field(..., description="Publication date")
    text: str = Field(..., description="Full paragraph text")
    sentences: List[PreprocessedSentence] = Field(default_factory=list)


class PreprocessedReport(BaseModel):
    """A fully preprocessed report preserving full document hierarchy."""

    report_id: str
    source: str
    publication_date: str
    title: str
    url: str = ""
    author: str = ""
    paragraphs: List[PreprocessedParagraph] = Field(default_factory=list)
    total_paragraphs: int = 0
    total_sentences: int = 0


class CTISegmenter:
    """Segments CTI reports into paragraphs and sentences while keeping track of provenance."""

    # Common abbreviations in English and cybersecurity to prevent false sentence breaks
    ABBREVIATIONS = {
        "e.g.", "i.e.", "u.s.", "u.k.", "dept.", "corp.", "inc.", "ltd.", "co.",
        "dr.", "mr.", "mrs.", "ms.", "vs.", "al.", "etc.", "approx.", "no.",
        "fig.", "cve-", "apt.", "sec.", "vol.", "gov.", "ref."
    }

    def __init__(self, cleaner: CTICleaner = None):
        self.cleaner = cleaner or CTICleaner()

    def segment_sentences(self, paragraph_text: str) -> List[tuple[str, int, int]]:
        """
        Split paragraph into sentences with character offsets.
        Returns list of (sentence_text, start_char, end_char).
        """
        if not paragraph_text.strip():
            return []

        # Split candidate sentences using punctuation + space/quote + uppercase
        # We protect abbreviations and known patterns
        candidates: List[tuple[str, int, int]] = []
        raw_text = paragraph_text
        length = len(raw_text)
        start = 0

        # Regex for candidate boundaries: . ! ? followed by space and Capital Letter
        boundary_regex = re.compile(r"([.!?]+)(['\"]?\s+)(?=[A-Z0-9\"])")

        matches = list(boundary_regex.finditer(raw_text))
        if not matches:
            text_strip = raw_text.strip()
            if text_strip:
                return [(text_strip, 0, len(text_strip))]
            return []

        for m in matches:
            punct_end = m.end(1)
            next_start = m.end()

            # Check if text before punct is an abbreviation
            token_before = raw_text[start:punct_end].split()[-1].lower() if raw_text[start:punct_end].split() else ""
            if token_before in self.ABBREVIATIONS or re.search(r"^[a-z]\.$", token_before):
                # Probably abbreviation, skip this boundary
                continue

            # Check if part of a version or CVE or IP address
            if re.search(r"\b\d+\.\d+$", raw_text[start:punct_end]):
                continue

            sentence = raw_text[start:punct_end].strip()
            if sentence:
                # Find exact start and end of trimmed sentence in raw_text
                s_idx = raw_text.find(sentence, start)
                candidates.append((sentence, s_idx, s_idx + len(sentence)))
            start = next_start

        # Trailing sentence
        if start < length:
            trailing = raw_text[start:].strip()
            if trailing:
                s_idx = raw_text.find(trailing, start)
                candidates.append((trailing, s_idx, s_idx + len(trailing)))

        return candidates

    def process_report(self, report: CTIReport) -> PreprocessedReport:
        """Clean and segment a raw CTI report into paragraphs and sentences."""
        cleaned_content = self.cleaner.clean(report.content)
        cleaned_title = self.cleaner.clean(report.title)

        raw_paragraphs = [p.strip() for p in cleaned_content.split("\n\n") if p.strip()]

        paragraphs: List[PreprocessedParagraph] = []
        global_s_idx = 0

        for p_idx, p_text in enumerate(raw_paragraphs):
            p_id = f"{report.report_id}::p{p_idx}"
            sentences_meta: List[PreprocessedSentence] = []

            sentence_tuples = self.segment_sentences(p_text)
            for s_idx, (s_text, start_c, end_c) in enumerate(sentence_tuples):
                s_id = f"{p_id}::s{s_idx}"
                sent_obj = PreprocessedSentence(
                    sentence_id=s_id,
                    paragraph_id=p_id,
                    report_id=report.report_id,
                    source=report.source,
                    publication_date=report.publication_date,
                    paragraph_index=p_idx,
                    sentence_index=s_idx,
                    global_sentence_index=global_s_idx,
                    text=s_text,
                    start_char=start_c,
                    end_char=end_c,
                )
                sentences_meta.append(sent_obj)
                global_s_idx += 1

            paragraph_obj = PreprocessedParagraph(
                paragraph_id=p_id,
                report_id=report.report_id,
                paragraph_index=p_idx,
                source=report.source,
                publication_date=report.publication_date,
                text=p_text,
                sentences=sentences_meta,
            )
            paragraphs.append(paragraph_obj)

        preprocessed_report = PreprocessedReport(
            report_id=report.report_id,
            source=report.source,
            publication_date=report.publication_date,
            title=cleaned_title,
            url=report.url or "",
            author=report.author or "",
            paragraphs=paragraphs,
            total_paragraphs=len(paragraphs),
            total_sentences=global_s_idx,
        )
        return preprocessed_report

    def process_all(self, reports: List[CTIReport]) -> List[PreprocessedReport]:
        """Preprocess a batch of CTI reports."""
        return [self.process_report(r) for r in reports]
