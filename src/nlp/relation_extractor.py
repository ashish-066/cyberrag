"""Cybersecurity Relationship Extraction Module.

Extracts core relationships:
- USES: ThreatActor/Malware -> Malware/Tool
- TARGETS: ThreatActor/Malware -> Organization/Location
- EXPLOITS: ThreatActor/Malware/Tool -> CVE
- USES_TECHNIQUE: ThreatActor/Malware -> Technique

Features:
- Clause-aware segmentation and intervening subject blocking (prevents cross-clause entanglement)
- True semantic trigger requirement (rejects naive co-occurrence and bare prepositions)
- Passive voice and directional syntactic parsing
- Negation detection and suppression
- Reporting vendor attribution filtering (prevents false TARGETS on security researchers)
- Strict provenance tracking on every single relation instance
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field

from src.preprocessor.segmenter import PreprocessedSentence, PreprocessedParagraph, PreprocessedReport
from .ner_engine import ExtractedEntity, CyberNEREngine

logger = logging.getLogger(__name__)


class ExtractedRelation(BaseModel):
    """A semantic relationship between two cybersecurity entities with evidence provenance."""

    relation_id: str = Field(..., description="Unique relationship identifier")
    relation_type: str = Field(..., description="Relationship type: USES, TARGETS, EXPLOITS, USES_TECHNIQUE")
    source_entity: str = Field(..., description="Canonical name of source entity")
    source_category: str = Field(..., description="Category of source entity")
    source_id: str = Field(..., description="Canonical ID of source entity")
    target_entity: str = Field(..., description="Canonical name of target entity")
    target_category: str = Field(..., description="Category of target entity")
    target_id: str = Field(..., description="Canonical ID of target entity")
    confidence: float = Field(default=0.85, description="Confidence score (0.0 to 1.0)")
    polarity: str = Field(default="positive", description="Assertion polarity: positive or negative")
    trigger: str = Field(..., description="Validated trigger verb or phrase establishing the relation")
    # Provenance fields
    evidence_text: str = Field(..., description="Sentence text providing factual grounding")
    sentence_id: str = Field(..., description="Provenance sentence ID")
    paragraph_id: str = Field(..., description="Provenance paragraph ID")
    report_id: str = Field(..., description="Provenance report ID")
    source: str = Field(..., description="Intel publication source")
    publication_date: str = Field(..., description="Intel publication date")


class CyberRelationExtractor:
    """Clause-aware, semantically grounded relationship extractor for CTI."""

    # Active verbal triggers establishing relationship
    ACTIVE_TRIGGERS: Dict[str, List[re.Pattern]] = {
        "USES": [
            re.compile(r"\b(?:used|uses|using|leveraged|leveraging|deployed|deploying|utilized|utilizing)\b", re.IGNORECASE),
            re.compile(r"\b(?:delivered|delivering|dropped|dropping|installed|installing|relied on|relying on)\b", re.IGNORECASE),
            re.compile(r"\b(?:armed with|weaponized|executed|executing|operators used|gang utilized|custom backdoor)\b", re.IGNORECASE),
            re.compile(r"\b(?:to deploy|to deliver|to drop|to install|to execute|to download)\b", re.IGNORECASE),
            re.compile(r"\b(?:download|downloads|downloaded|downloading|distribute|distributes|distributed|distributing)\b", re.IGNORECASE),
        ],
        "TARGETS": [
            re.compile(r"\b(?:target|targets|targeted|targeting|attack|attacks|attacked|attacking|focused on|focusing on)\b", re.IGNORECASE),
            re.compile(r"\b(?:victimizing|victims in|directed against|compromise|compromises|compromised|compromising|infiltrate|infiltrates|infiltrated|infiltrating)\b", re.IGNORECASE),
            re.compile(r"\b(?:infect|infects|infected|infecting|disrupt|disrupts|disrupted|disrupting|hit|hits|hitting)\b", re.IGNORECASE),
            re.compile(r"\b(?:launched destructive campaigns against|launched campaigns against|operations against)\b", re.IGNORECASE),
            re.compile(r"\b(?:intrusions into|intrusions against|channels into|exfiltrate data from)\b", re.IGNORECASE),
            re.compile(r"\b(?:impair|impairs|impaired|impairing|deliver(?:ed|ing)?\s+to|disseminat(?:e|ed|ing)\s+(?:.*?\s+)?across)\b", re.IGNORECASE),
        ],
        "EXPLOITS": [
            re.compile(r"\b(?:exploited|exploiting|exploits|abused|abusing|leveraged vulnerability|leveraging vulnerability)\b", re.IGNORECASE),
            re.compile(r"\b(?:utilize|utilizes|utilized|utilizing|leverage|leveraged|leveraging|exploit|exploited|exploits|exploiting)\s+(?:.*?vulnerabilit(?:y|ies)|flaws?|zero-days?)\b", re.IGNORECASE),
            re.compile(r"\b(?:vulnerability in|flaw in|zero-day in|zero-day vulnerabilities in|remote code execution in|targeting vulnerability|vulnerabilities attached to|vulnerability attached to|vulnerabilities like)\b", re.IGNORECASE),
        ],
        "USES_TECHNIQUE": [
            re.compile(r"\b(?:employed|employing|employs|utilized|utilizing|executed|executing|conducted|conducting)\b", re.IGNORECASE),
            re.compile(r"\b(?:relied on|relying on|leveraged|leveraging|via|through|employed technique|utilize|using|used|uses)\b", re.IGNORECASE),
        ],
    }

    # Passive voice trigger patterns
    PASSIVE_TRIGGERS: Dict[str, List[re.Pattern]] = {
        "USES": [
            re.compile(r"\b(?:was deployed by|were deployed by|deployed by|was used by|used by|was dropped by|was distributed by|distributed by|downloaded by)\b", re.IGNORECASE),
        ],
        "TARGETS": [
            re.compile(r"\b(?:was targeted by|were targeted by|targeted by|was attacked by|attacked by|compromised by)\b", re.IGNORECASE),
        ],
        "EXPLOITS": [
            re.compile(r"\b(?:was exploited by|were exploited by|was actively exploited by|exploited by|leveraged by)\b", re.IGNORECASE),
        ],
        "USES_TECHNIQUE": [
            re.compile(r"\b(?:was executed by|were executed by|executed by|conducted by)\b", re.IGNORECASE),
        ],
    }

    # Negation cues
    NEGATION_PATTERNS = [
        re.compile(r"\b(?:did not|does not|do not|never|no evidence|no indications?|failed to|without deploying|without using)\b", re.IGNORECASE),
    ]

    # Reporting attribution patterns (identifies vendors reporting rather than victims)
    REPORTING_VENDOR_PATTERNS = [
        re.compile(r"\b(?:researchers (?:at|from|identified|observed)|analysts (?:at|from)|according to|reported by|investigation by|blog by)\s+([A-Za-z0-9_ -]+)", re.IGNORECASE),
        re.compile(r"([A-Za-z0-9_ -]+)\s+(?:researchers|analysts|threat intelligence|uncovered|observed|identified|tracked|issued|published)", re.IGNORECASE),
    ]

    # Clause boundary delimiters
    CLAUSE_DELIMITERS = re.compile(r";|\b(?:while|whereas|although|however|but)\b", re.IGNORECASE)

    ALLOWED_SCHEMAS = {
        "USES": {
            ("ThreatActor", "Malware"),
            ("ThreatActor", "Tool"),
            ("Malware", "Tool"),
            ("Malware", "Malware"),
        },
        "TARGETS": {
            ("ThreatActor", "Organization"),
            ("ThreatActor", "Location"),
            ("Malware", "Organization"),
            ("Malware", "Location"),
        },
        "EXPLOITS": {
            ("ThreatActor", "CVE"),
            ("Malware", "CVE"),
            ("Tool", "CVE"),
        },
        "USES_TECHNIQUE": {
            ("ThreatActor", "Technique"),
            ("Malware", "Technique"),
        },
    }

    def __init__(self, ner_engine: Optional[CyberNEREngine] = None):
        self.ner_engine = ner_engine or CyberNEREngine()

    def _is_negated(self, text_segment: str) -> bool:
        """Check if negation cues appear in the governing segment."""
        for pat in self.NEGATION_PATTERNS:
            if pat.search(text_segment):
                return True
        return False

    def _is_reporting_source(self, text: str, org_name: str, org_start: int) -> bool:
        """Check if organization mention is acting as reporting source rather than victim target."""
        window_start = max(0, org_start - 45)
        window_end = min(len(text), org_start + len(org_name) + 45)
        window = text[window_start:window_end]

        for pat in self.REPORTING_VENDOR_PATTERNS:
            match = pat.search(window)
            if match:
                vendor_matched = match.group(1).strip().lower()
                clean_org = org_name.strip().lower()
                if clean_org in vendor_matched or vendor_matched in clean_org:
                    return True
        return False

    def _has_intervening_clause_break(self, text: str, start_pos: int, end_pos: int) -> bool:
        """Check if a contrasting clause boundary breaks the syntactic relationship."""
        segment = text[start_pos:end_pos]
        return bool(self.CLAUSE_DELIMITERS.search(segment))

    def _find_active_trigger_between(self, text: str, rel_type: str, start_pos: int, end_pos: int) -> Optional[Tuple[str, float]]:
        """Look for an active trigger verb strictly between subject and object."""
        segment = text[start_pos:end_pos]
        for pattern in self.ACTIVE_TRIGGERS.get(rel_type, []):
            match = pattern.search(segment)
            if match:
                return match.group(0), 0.95
        return None

    def _find_passive_trigger_between(self, text: str, rel_type: str, start_pos: int, end_pos: int) -> Optional[Tuple[str, float]]:
        """Look for a passive trigger strictly between object and subject."""
        segment = text[start_pos:end_pos]
        for pattern in self.PASSIVE_TRIGGERS.get(rel_type, []):
            match = pattern.search(segment)
            if match:
                return match.group(0), 0.92
        return None

    def extract_from_sentence(
        self,
        sentence: PreprocessedSentence,
        entities: List[ExtractedEntity],
        context_actor: Optional[ExtractedEntity] = None,
    ) -> List[ExtractedRelation]:
        """
        Extract relationships from entities within a single sentence enforcing semantic quality.
        Rejects arbitrary co-occurrence without valid action triggers.
        """
        relations: List[ExtractedRelation] = []
        text = sentence.text

        # Deduplicate entities by (category, canonical_name)
        unique_entities: Dict[str, ExtractedEntity] = {}
        for ent in entities:
            key = f"{ent.category}::{ent.canonical_name}"
            if key not in unique_entities:
                unique_entities[key] = ent
        ent_list = list(unique_entities.values())

        actors_in_sent = [e for e in ent_list if e.category == "ThreatActor"]

        # Sentence-level negation check
        sentence_has_negation = self._is_negated(text)

        # 1. Pairwise syntactic evaluation
        for i, ent_a in enumerate(ent_list):
            for j, ent_b in enumerate(ent_list):
                if i == j:
                    continue
                pair_type = (ent_a.category, ent_b.category)

                for rel_type, allowed_pairs in self.ALLOWED_SCHEMAS.items():
                    if pair_type not in allowed_pairs:
                        continue

                    # Reporting vendor check: Only check Organization entities, never Malware or ThreatActor
                    if ent_a.category == "Organization" and self._is_reporting_source(text, ent_a.text, ent_a.start_char):
                        continue
                    if rel_type == "TARGETS" and ent_b.category == "Organization":
                        if self._is_reporting_source(text, ent_b.text, ent_b.start_char):
                            continue

                    trigger_found = None
                    confidence = 0.0

                    # Check for intervening actor between ent_a and ent_b
                    # If another Threat Actor is situated between ent_a and ent_b, block ent_a from claiming ent_b
                    if ent_a.category == "ThreatActor":
                        intervening_actors = [
                            other for other in actors_in_sent
                            if other != ent_a and min(ent_a.start_char, ent_b.start_char) < other.start_char < max(ent_a.start_char, ent_b.start_char)
                        ]
                        if intervening_actors:
                            continue

                    # Coordinate action clause check:
                    # If ent_a is not a ThreatActor (e.g. Malware), but an earlier ThreatActor is present,
                    # and the trigger between ent_a and ent_b is introduced by coordinate conjunction (e.g. "and utilized", "and exploited"),
                    # the governing agent is the earlier ThreatActor, NOT ent_a.
                    if ent_a.category != "ThreatActor" and actors_in_sent:
                        earlier_actors = [a for a in actors_in_sent if a.start_char < ent_a.start_char]
                        if earlier_actors:
                            inter_segment = text[ent_a.end_char:ent_b.start_char]
                            if re.search(r"^\s*(?:,|and|as well as)\s+(?:also\s+)?(?:utilized|utilizing|used|using|deployed|deploying|exploited|exploiting|employed|conducted|targeted)\b", inter_segment, re.IGNORECASE):
                                continue

                    # Check for intervening clause break
                    min_pos = min(ent_a.end_char, ent_b.end_char)
                    max_pos = max(ent_a.start_char, ent_b.start_char)
                    if self._has_intervening_clause_break(text, min_pos, max_pos):
                        # Crosses a clause boundary (e.g. "While Volt Typhoon used X, APT29 used Y")
                        # Disallow binding across clause boundary
                        continue

                    # A. Active voice: Subject appears BEFORE Object
                    if ent_a.start_char < ent_b.start_char:
                        active_res = self._find_active_trigger_between(
                            text, rel_type, ent_a.end_char, ent_b.start_char
                        )
                        if active_res:
                            trigger_found, confidence = active_res

                    # B. Passive voice: Object appears BEFORE Subject
                    elif ent_b.start_char < ent_a.start_char:
                        # Intervening entity check between Object and Subject
                        intervening_ents = [
                            e for e in ent_list
                            if e != ent_a and e != ent_b and min(ent_b.end_char, ent_a.end_char) < e.start_char < max(ent_b.start_char, ent_a.start_char)
                        ]
                        if intervening_ents:
                            continue
                        pass_res = self._find_passive_trigger_between(
                            text, rel_type, ent_b.end_char, ent_a.start_char
                        )
                        if pass_res:
                            trigger_found, confidence = pass_res

                    # Strict Quality Filter:
                    if trigger_found and confidence >= 0.75:
                        is_negated = False
                        # Localized negation check focused on the trigger/verb predicate
                        trigger_idx = text.find(trigger_found, min(ent_a.start_char, ent_b.start_char))
                        if trigger_idx != -1:
                            window_start = max(0, trigger_idx - 30)
                            window_end = min(len(text), trigger_idx + len(trigger_found) + 15)
                            trigger_window = text[window_start:window_end]
                            if self._is_negated(trigger_window):
                                is_negated = True
                        else:
                            span_text = text[min(ent_a.start_char, ent_b.start_char):max(ent_a.end_char, ent_b.end_char)]
                            if self._is_negated(span_text):
                                is_negated = True

                        rel_polarity = "negative" if is_negated else "positive"
                        rel_id = f"rel::{ent_a.canonical_name}::{rel_type}::{ent_b.canonical_name}::{sentence.sentence_id}::{rel_polarity}"
                        relations.append(
                            ExtractedRelation(
                                relation_id=rel_id,
                                relation_type=rel_type,
                                source_entity=ent_a.canonical_name,
                                source_category=ent_a.category,
                                source_id=ent_a.canonical_id,
                                target_entity=ent_b.canonical_name,
                                target_category=ent_b.category,
                                target_id=ent_b.canonical_id,
                                confidence=confidence,
                                polarity=rel_polarity,
                                trigger=trigger_found,
                                evidence_text=sentence.text,
                                sentence_id=sentence.sentence_id,
                                paragraph_id=sentence.paragraph_id,
                                report_id=sentence.report_id,
                                source=sentence.source,
                                publication_date=sentence.publication_date,
                            )
                        )

        # 2. Contextual Paragraph Coreference
        # Only when no explicit actors in sentence AND no negation
        if not actors_in_sent and context_actor and not sentence_has_negation:
            for ent_target in ent_list:
                pair_type = (context_actor.category, ent_target.category)

                for rel_type, allowed_pairs in self.ALLOWED_SCHEMAS.items():
                    if pair_type not in allowed_pairs:
                        continue

                    if rel_type == "TARGETS" and ent_target.category == "Organization":
                        if self._is_reporting_source(text, ent_target.text, ent_target.start_char):
                            continue

                    # Require explicit active trigger in current sentence
                    for pat in self.ACTIVE_TRIGGERS.get(rel_type, []):
                        match = pat.search(text)
                        if match and match.start() <= ent_target.start_char:
                            rel_id = f"rel::{context_actor.canonical_name}::{rel_type}::{ent_target.canonical_name}::{sentence.sentence_id}"
                            relations.append(
                                ExtractedRelation(
                                    relation_id=rel_id,
                                    relation_type=rel_type,
                                    source_entity=context_actor.canonical_name,
                                    source_category=context_actor.category,
                                    source_id=context_actor.canonical_id,
                                    target_entity=ent_target.canonical_name,
                                    target_category=ent_target.category,
                                    target_id=ent_target.canonical_id,
                                    confidence=0.82,
                                    polarity="positive",
                                    trigger=f"contextual:{match.group(0)}",
                                    evidence_text=sentence.text,
                                    sentence_id=sentence.sentence_id,
                                    paragraph_id=sentence.paragraph_id,
                                    report_id=sentence.report_id,
                                    source=sentence.source,
                                    publication_date=sentence.publication_date,
                                )
                            )
                            break

        return relations

    def extract_from_paragraph(
        self,
        paragraph: PreprocessedParagraph,
        paragraph_entities: List[ExtractedEntity],
    ) -> List[ExtractedRelation]:
        """Extract relationships from a paragraph tracking dominant threat actor context."""
        relations: List[ExtractedRelation] = []

        actors_in_p = [e for e in paragraph_entities if e.category == "ThreatActor"]
        dominant_actor = actors_in_p[0] if actors_in_p else None

        sent_entities_map: Dict[str, List[ExtractedEntity]] = {}
        for ent in paragraph_entities:
            sent_entities_map.setdefault(ent.sentence_id, []).append(ent)

        for sent in paragraph.sentences:
            s_ents = sent_entities_map.get(sent.sentence_id, [])
            s_actors = [e for e in s_ents if e.category == "ThreatActor"]
            if s_actors:
                dominant_actor = s_actors[0]

            s_rels = self.extract_from_sentence(
                sent,
                s_ents,
                context_actor=dominant_actor,
            )
            relations.extend(s_rels)

        return relations

    def extract_from_report(self, report: PreprocessedReport) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """Extract entities and high-quality relationships from a CTI report."""
        report_entities: List[ExtractedEntity] = []
        report_relations: List[ExtractedRelation] = []

        for paragraph in report.paragraphs:
            p_entities: List[ExtractedEntity] = []
            for sent in paragraph.sentences:
                s_entities = self.ner_engine.extract_from_sentence(sent)
                p_entities.extend(s_entities)
            report_entities.extend(p_entities)

            p_relations = self.extract_from_paragraph(paragraph, p_entities)
            report_relations.extend(p_relations)

        # Deduplicate relationships by (source_entity, relation_type, target_entity, sentence_id, polarity)
        seen_keys = set()
        deduped_relations = []
        for r in report_relations:
            key = (r.source_entity, r.relation_type, r.target_entity, r.sentence_id, r.polarity)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_relations.append(r)

        return report_entities, deduped_relations

    def extract_all(
        self, reports: List[PreprocessedReport]
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """Extract all entities and relationships across all reports."""
        all_entities: List[ExtractedEntity] = []
        all_relations: List[ExtractedRelation] = []

        for report in reports:
            entities, relations = self.extract_from_report(report)
            all_entities.extend(entities)
            all_relations.extend(relations)

        logger.info(
            f"Extracted {len(all_entities)} entities and {len(all_relations)} relations across {len(reports)} reports."
        )
        return all_entities, all_relations
