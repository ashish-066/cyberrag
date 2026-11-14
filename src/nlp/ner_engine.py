"""Cybersecurity Named Entity Recognition (NER) Engine.

Extracts Threat Actors, Malware, Tools, CVEs, Techniques, Organizations, and Locations
from preprocessed CTI sentences, linking and disambiguating them against canonical registries.
Supports hybrid rule-based and contextual ML token extraction.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field

from src.preprocessor.segmenter import PreprocessedSentence, PreprocessedReport
from .gazetteer import (
    THREAT_ACTORS,
    MALWARE,
    TOOLS,
    KNOWN_CVES,
    TECHNIQUES,
    ORGANIZATIONS,
    LOCATIONS,
)
from .normalizer import EntityNormalizer
from .ml_ner import CyberMLNER

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """Structured Cybersecurity Entity extracted from CTI text with full provenance."""

    entity_id: str = Field(..., description="Unique entity ID (e.g. Category::CanonicalName)")
    text: str = Field(..., description="Raw surface text extracted from sentence")
    surface_text: str = Field(default="", description="Original verbatim surface token")
    category: str = Field(..., description="Entity Category (ThreatActor, Malware, Tool, CVE, Technique, Organization, Location)")
    canonical_name: str = Field(..., description="Canonical linked name")
    canonical_id: str = Field(..., description="Canonical ID (e.g. MITRE G-code, CVE ID)")
    start_char: int = Field(..., description="Start character offset in sentence")
    end_char: int = Field(..., description="End character offset in sentence")
    confidence: float = Field(default=0.95, description="Confidence score (0.0 to 1.0)")
    is_ambiguous: bool = Field(default=False, description="Whether entity matched multiple canonical interpretations")
    ambiguity_candidates: List[str] = Field(default_factory=list, description="Alternative candidate IDs if ambiguous")
    # Provenance fields
    sentence_id: str = Field(..., description="Source sentence ID")
    paragraph_id: str = Field(..., description="Source paragraph ID")
    report_id: str = Field(..., description="Source CTI report ID")
    source: str = Field(..., description="Publisher source")
    publication_date: str = Field(..., description="Publication date of the report")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional linked attributes")


class CyberNEREngine:
    """Cybersecurity NER extractor combining regex rules, compiled gazetteers, and contextual ML."""

    # Regex patterns
    CVE_REGEX = re.compile(r"\bCVE[-_ ](\d{4})[-_ ](\d{4,7})\b", re.IGNORECASE)
    TECHNIQUE_ID_REGEX = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)

    def __init__(
        self,
        normalizer: Optional[EntityNormalizer] = None,
        enable_ml: bool = False,
        ml_model: Optional[CyberMLNER] = None,
    ):
        self.normalizer = normalizer or EntityNormalizer()
        self.enable_ml = enable_ml
        self.ml_model = ml_model or (CyberMLNER() if enable_ml else None)
        self._compiled_patterns: List[Tuple[re.Pattern, str, str, Dict]] = []
        self._build_compiled_patterns()

    def _build_compiled_patterns(self):
        """Compile regex patterns from gazetteer sorted by length descending."""
        entries = []

        # 1. Threat Actors
        for canon, data in THREAT_ACTORS.items():
            for alias in data["aliases"]:
                entries.append((alias, "ThreatActor", canon, data))

        # 2. Malware
        for canon, data in MALWARE.items():
            for alias in data["aliases"]:
                entries.append((alias, "Malware", canon, data))

        # 3. Tools
        for canon, data in TOOLS.items():
            for alias in data["aliases"]:
                entries.append((alias, "Tool", canon, data))

        # 4. Known CVE named aliases
        for canon, data in KNOWN_CVES.items():
            for alias in data["aliases"]:
                if not alias.upper().startswith("CVE-"):
                    entries.append((alias, "CVE", canon, data))

        # 5. MITRE Techniques
        for tech_id, data in TECHNIQUES.items():
            canon = f"{tech_id} - {data['name']}"
            for alias in data["aliases"]:
                # Generic noun: not sufficient evidence of a MITRE technique
                if (
                    tech_id in ("T1587.001", "T1588.001")
                    and alias.strip().lower() == "malware"
                ):
                    continue
                entries.append((alias, "Technique", canon, data))

        # 6. Organizations
        for canon, data in ORGANIZATIONS.items():
            for alias in data["aliases"]:
                entries.append((alias, "Organization", canon, data))

        # 7. Locations
        for canon, data in LOCATIONS.items():
            for alias in data["aliases"]:
                entries.append((alias, "Location", canon, data))

        # Sort longest surface string first to prevent substring collisions
        entries.sort(key=lambda x: len(x[0]), reverse=True)

        for alias, category, canon_name, meta in entries:
            escaped = re.escape(alias)
            pattern_str = rf"(?<![A-Za-z0-9_-]){escaped}(?![A-Za-z0-9_-])"
            compiled = re.compile(pattern_str, re.IGNORECASE)
            self._compiled_patterns.append((compiled, category, canon_name, meta))

    @staticmethod
    def is_non_entity_artifact(text: str) -> bool:
        """Reject malformed markdown, MD5 hashes, and URL artifacts."""
        value = text.strip()
        if re.fullmatch(r"[-=_]{2,}", value):
            return True
        if re.fullmatch(r"[A-Fa-f0-9]{32}-?", value):
            return True
        if re.search(r"^(https?://|www\.)", value, re.I):
            return True
        if re.search(r"\.html?$", value, re.I) and len(value) > 35:
            return True
        return False

    @staticmethod
    def _mask_url_artifacts(text: str) -> str:
        """Mask URLs and Markdown link destinations without changing offsets."""
        masked = list(text)

        patterns = [
            # Markdown image destinations (with whitespace tolerance inside parentheses)
            r'!\[[^\]]*\]\s*\(\s*[^)]+\)',
            # Markdown link destinations (with whitespace tolerance inside parentheses)
            r'\[[^\]]+\]\s*\(\s*[^)]+\)',
            r'\]\s*\(\s*[^)]+\)',
            # Base64 data URIs
            r'data:image/[^,\s)]+;base64,[A-Za-z0-9+/=\s]+',
            # Complete URLs
            r'(?i)\b(?:https?://|www\.)[^\s<>"\']+',
        ]

        for pat_str in patterns:
            pat = re.compile(pat_str, re.IGNORECASE)
            for match in pat.finditer(text):
                for i in range(match.start(), match.end()):
                    if masked[i] not in "\r\n":
                        masked[i] = " "

        return "".join(masked)

    @staticmethod
    def _is_ambiguous_alias_valid(
        text: str,
        start: int,
        end: int,
        category: str,
        canonical_name: str
    ) -> bool:
        """Reject ambiguous short aliases without removing valid MITRE entities."""
        surface = text[start:end]
        lower = surface.lower()

        # United States abbreviation
        if category == "Location" and canonical_name == "United States":
            if lower == "us":
                return surface == "US"

        # Windows Net utility
        if category == "Tool" and canonical_name == "Net":
            if lower == "net":
                before = text[max(0, start - 30):start].lower()
                after = text[end:end + 30].lower()

                command_context = (
                    any(x in before for x in ("command ", "execute ", "running "))
                    or any(x in after for x in (
                        ".exe", " use ", " user ", " group ", " share "
                    ))
                )

                return surface == "Net" and command_context

        # at scheduling utility or technique (S0110 / T1053.002)
        if (category in ("Tool", "Technique")) and (canonical_name == "at" or "at" in canonical_name.lower().split()):
            if lower == "at":
                before = text[max(0, start - 30):start].lower()
                after = text[end:end + 30].lower()

                command_context = (
                    any(x in before for x in ("command ", "execute ", "running "))
                    or any(x in after for x in (
                        ".exe", " command", " job", " utility"
                    ))
                )

                return command_context

        return True

    def extract_from_sentence(self, sentence: PreprocessedSentence) -> List[ExtractedEntity]:
        """Extract and normalize all cybersecurity entities from a single sentence."""
        text = sentence.text
        raw_spans: List[Dict[str, Any]] = []

        # 1. Extract CVEs via standard regex
        for match in self.CVE_REGEX.finditer(text):
            raw_cve = match.group(0)
            norm_res = self.normalizer.normalize(raw_cve, category="CVE", context=text)
            raw_spans.append({
                "start": match.start(),
                "end": match.end(),
                "text": raw_cve,
                "surface_text": raw_cve,
                "category": "CVE",
                "canonical_name": norm_res["canonical_name"],
                "canonical_id": norm_res["canonical_id"],
                "confidence": 0.99,
                "is_ambiguous": norm_res.get("is_ambiguous", False),
                "ambiguity_candidates": norm_res.get("ambiguity_candidates", []),
                "metadata": norm_res.get("metadata", {}),
            })

        # 2. Extract Technique IDs via regex
        for match in self.TECHNIQUE_ID_REGEX.finditer(text):
            raw_tech = match.group(0).upper()
            norm_res = self.normalizer.normalize(raw_tech, category="Technique", context=text)
            raw_spans.append({
                "start": match.start(),
                "end": match.end(),
                "text": raw_tech,
                "surface_text": raw_tech,
                "category": "Technique",
                "canonical_name": norm_res["canonical_name"],
                "canonical_id": norm_res["canonical_id"],
                "confidence": 0.98,
                "is_ambiguous": norm_res.get("is_ambiguous", False),
                "ambiguity_candidates": norm_res.get("ambiguity_candidates", []),
                "metadata": norm_res.get("metadata", {}),
            })

        # 3. Extract gazetteer entities (using masked URL copy to prevent false positives)
        matching_text = self._mask_url_artifacts(text)

        for pattern, category, canon_name, meta in self._compiled_patterns:
            for match in pattern.finditer(matching_text):
                if not self._is_ambiguous_alias_valid(
                    text,
                    match.start(),
                    match.end(),
                    category,
                    canon_name
                ):
                    continue

                raw_text = text[match.start():match.end()]

                norm_res = self.normalizer.normalize(
                    raw_text,
                    category=category,
                    context=text
                )

                raw_spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": raw_text,
                    "surface_text": raw_text,
                    "category": norm_res.get("category", category),
                    "canonical_name": norm_res.get("canonical_name", canon_name),
                    "canonical_id": norm_res.get(
                        "canonical_id",
                        meta.get("canonical_id", canon_name)
                    ),
                    "confidence": 0.95,
                    "is_ambiguous": norm_res.get("is_ambiguous", False),
                    "ambiguity_candidates": norm_res.get("ambiguity_candidates", []),
                    "metadata": norm_res.get("metadata", meta)
                })

        # 4. Optional Contextual ML Model Predictions (for novel entities)
        if self.enable_ml and self.ml_model and self.ml_model.is_trained:
            ml_preds = self.ml_model.predict_sentence(sentence)
            for mp in ml_preds:
                raw_spans.append({
                    "start": mp["start_char"],
                    "end": mp["end_char"],
                    "text": mp["text"],
                    "surface_text": mp["text"],
                    "category": mp["category"],
                    "canonical_name": mp["canonical_name"],
                    "canonical_id": mp["canonical_id"],
                    "confidence": mp["confidence"],
                    "is_ambiguous": False,
                    "ambiguity_candidates": [],
                    "metadata": {"ml_model": "CyberMLNER"},
                })

        # 5. Overlap resolution (longest match & highest confidence preference)
        raw_spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"]), -s["confidence"]))

        filtered_spans: List[Dict[str, Any]] = []
        occupied_intervals: List[Tuple[int, int]] = []

        for span in raw_spans:
            if self.is_non_entity_artifact(span["surface_text"]):
                continue
            s_start, s_end = span["start"], span["end"]
            overlaps = False
            for o_start, o_end in occupied_intervals:
                if max(s_start, o_start) < min(s_end, o_end):
                    overlaps = True
                    break
            if not overlaps:
                occupied_intervals.append((s_start, s_end))
                filtered_spans.append(span)

        # 6. Convert to ExtractedEntity models
        entities: List[ExtractedEntity] = []
        for s in filtered_spans:
            entity_id = f"{s['category']}::{s['canonical_name']}"
            entities.append(
                ExtractedEntity(
                    entity_id=entity_id,
                    text=s["text"],
                    surface_text=s["surface_text"],
                    category=s["category"],
                    canonical_name=s["canonical_name"],
                    canonical_id=s["canonical_id"],
                    start_char=s["start"],
                    end_char=s["end"],
                    confidence=s["confidence"],
                    is_ambiguous=s.get("is_ambiguous", False),
                    ambiguity_candidates=s.get("ambiguity_candidates", []),
                    sentence_id=sentence.sentence_id,
                    paragraph_id=sentence.paragraph_id,
                    report_id=sentence.report_id,
                    source=sentence.source,
                    publication_date=sentence.publication_date,
                    metadata=s["metadata"],
                )
            )

        entities.sort(key=lambda e: e.start_char)
        return entities

    def extract_from_report(self, report: PreprocessedReport) -> List[ExtractedEntity]:
        """Extract entities from all sentences of a preprocessed report."""
        report_entities: List[ExtractedEntity] = []
        for p in report.paragraphs:
            for s in p.sentences:
                entities = self.extract_from_sentence(s)
                report_entities.extend(entities)
        return report_entities

    def extract_all(self, reports: List[PreprocessedReport]) -> List[ExtractedEntity]:
        """Extract entities across a list of preprocessed reports."""
        all_entities: List[ExtractedEntity] = []
        for r in reports:
            all_entities.extend(self.extract_from_report(r))
        logger.info(f"Extracted total {len(all_entities)} entity mentions across {len(reports)} reports.")
        return all_entities
