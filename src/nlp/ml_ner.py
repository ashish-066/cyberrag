"""Machine Learning & Contextual Cybersecurity NER Module.

Extracts cybersecurity entities using contextual token features and statistical classification
(scikit-learn), enabling identification of novel/unseen threat actors, malware, and tools
not present in static gazetteers.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from src.preprocessor.segmenter import PreprocessedSentence

logger = logging.getLogger(__name__)


class MLNERToken(BaseModel):
    """Token representation with extracted contextual features."""
    text: str
    start: int
    end: int
    features: Dict[str, Any]


class CyberMLNER:
    """Contextual Machine Learning Named Entity Recognizer for Cybersecurity."""

    CYBER_SUFFIXES = (
        "rat", "wiper", "bot", "locker", "lock", "strike", "typhoon", "bear",
        "group", "tempest", "spider", "kitten", "chollima", "blizzard", "panda",
        "stealer", "trojan", "loader", "c2", "proxy", "exec"
    )

    CYBER_PREFIXES = (
        "cve-", "apt", "ta-", "unc", "msdt", "kb"
    )

    TAG_TO_CATEGORY = {
        "GROUP": "ThreatActor",
        "ACTOR": "ThreatActor",
        "THREATACTOR": "ThreatActor",
        "MALWARE": "Malware",
        "TOOL": "Tool",
        "TECHNIQUE": "Technique",
        "TACTIC": "Technique",
        "ORG": "Organization",
        "ORGANIZATION": "Organization",
        "SECTOR": "Organization",
        "LOC": "Location",
        "LOCATION": "Location",
        "VULN": "CVE",
        "VULNERABILITY": "CVE",
        "CVE": "CVE",
    }

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self._init_model()

    def _init_model(self):
        """Initialize the vectorizer and classifier pipeline."""
        self.pipeline = Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", LogisticRegression(max_iter=500, C=1.5, class_weight="balanced")),
        ])

    def extract_token_features(self, tokens: List[Tuple[str, int, int]], i: int) -> Dict[str, Any]:
        """Extract rich orthographic, morphological, and contextual window features."""
        word, start, end = tokens[i]
        lower_word = word.lower()

        features = {
            "bias": 1.0,
            "word.lower": lower_word,
            "word.isupper": word.isupper(),
            "word.istitle": word.istitle(),
            "word.isdigit": word.isdigit(),
            "word.has_hyphen": "-" in word,
            "word.has_digit": any(c.isdigit() for c in word),
            "word.prefix_cve": lower_word.startswith("cve-"),
            "word.prefix_mitre": bool(re.match(r"^t\d{4}", lower_word)),
            "word.suffix_cyber": any(lower_word.endswith(sfx) for sfx in self.CYBER_SUFFIXES),
            "word.prefix_cyber": any(lower_word.startswith(pfx) for pfx in self.CYBER_PREFIXES),
            "word.length": len(word),
        }

        # Contextual window features: Previous token (-1)
        if i > 0:
            prev_word = tokens[i - 1][0]
            prev_lower = prev_word.lower()
            features["-1:word.lower"] = prev_lower
            features["-1:word.istitle"] = prev_word.istitle()
            features["-1:is_trigger_actor"] = prev_lower in ("actor", "group", "threat", "adversary", "gang", "unit")
            features["-1:is_trigger_malware"] = prev_lower in ("malware", "ransomware", "trojan", "wiper", "backdoor", "payload")
            features["-1:is_trigger_tool"] = prev_lower in ("tool", "utility", "script", "framework")
            features["-1:is_trigger_target"] = prev_lower in ("targeted", "against", "across", "compromised", "in")
            features["-1:is_trigger_exploit"] = prev_lower in ("exploited", "vulnerability", "flaw", "cve")
        else:
            features["BOS"] = True

        # Contextual window features: Next token (+1)
        if i < len(tokens) - 1:
            next_word = tokens[i + 1][0]
            next_lower = next_word.lower()
            features["+1:word.lower"] = next_lower
            features["+1:word.istitle"] = next_word.istitle()
            features["+1:is_actor_noun"] = next_lower in ("group", "gang", "operators", "affiliates", "actor")
            features["+1:is_malware_noun"] = next_lower in ("ransomware", "wiper", "trojan", "malware", "backdoor")
            features["+1:is_tool_noun"] = next_lower in ("utility", "framework", "tool", "binary")
        else:
            features["EOS"] = True

        return features

    def tokenize_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenize sentence preserving exact character start/end offsets."""
        token_pattern = re.compile(r"[A-Za-z0-9_.-]+|[^\s\w]")
        tokens = []
        for match in token_pattern.finditer(text):
            tok_text = match.group(0).strip(".,!?;:")
            if tok_text:
                tokens.append((tok_text, match.start(), match.start() + len(tok_text)))
        return tokens

    def train_on_annotated_sentences(self, annotated_data: List[Dict[str, Any]]) -> None:
        """Train the classifier on annotated ground-truth sentences using aligned token-level BIO tags."""
        X_features: List[Dict[str, Any]] = []
        y_labels: List[str] = []

        for sample in annotated_data:
            # 1. Direct token-level BIO alignment
            raw_tokens = sample.get("tokens")
            tags = sample.get("all_tags") or sample.get("bio_tags")

            if raw_tokens and tags and len(raw_tokens) == len(tags):
                text = sample.get("text", "")
                token_tuples = []
                cursor = 0
                for t in raw_tokens:
                    idx = text.find(t, cursor)
                    if idx != -1:
                        token_tuples.append((t, idx, idx + len(t)))
                        cursor = idx + len(t)
                    else:
                        token_tuples.append((t, cursor, cursor + len(t)))
                        cursor += len(t) + 1

                for i in range(len(raw_tokens)):
                    feat = self.extract_token_features(token_tuples, i)
                    tag = tags[i]
                    label = "O"
                    if tag and tag != "O":
                        tag_type = tag.split("-", 1)[1].upper() if "-" in tag else tag.upper()
                        label = self.TAG_TO_CATEGORY.get(tag_type, "O")
                    X_features.append(feat)
                    y_labels.append(label)
                continue

            # 2. Fallback for unsegmented samples without token-level BIO tags
            text = sample.get("text", "")
            tokens = self.tokenize_with_spans(text)
            if not tokens:
                continue

            ents = sample.get("ground_truth_entities", [])
            for i, (tok_text, s_start, s_end) in enumerate(tokens):
                feat = self.extract_token_features(tokens, i)
                label = "O"
                tok_lower = tok_text.lower()
                for ent in ents:
                    e_canon = ent.get("canonical_name", "").lower()
                    e_surface = ent.get("surface_text", "").lower()
                    if tok_lower == e_canon or tok_lower == e_surface:
                        label = ent.get("category", "O")
                        break

                X_features.append(feat)
                y_labels.append(label)

        if len(set(y_labels)) > 1:
            self.pipeline.fit(X_features, y_labels)
            self.is_trained = True
            logger.info(f"CyberMLNER trained on {len(X_features)} tokens across {len(annotated_data)} sentences.")
        else:
            logger.warning("Insufficient label diversity to train ML NER model.")

    GENERIC_NON_ENTITIES = {
        "sector", "sectors", "enterprises", "enterprise", "water", "chains", "chain",
        "recovery", "firms", "firm", "organizations", "organization", "systems", "system",
        "entities", "entity", "data", "encrypted", "affiliates", "affiliate", "group",
        "operators", "operator", "actors", "actor", "threat", "adversary", "malware",
        "trojan", "tool", "vulnerability", "living-off-the-land", "utilities", "utility",
        "increase", "activity", "operations", "campaigns", "campaign", "indicators", "guidance"
    }

    SECURITY_VENDORS = {
        "crowdstrike", "mandiant", "cisa", "sentinelone", "microsoft", "palo alto",
        "palo alto networks", "unit 42", "symantec", "fireeye", "kaspersky", "eset", "proofpoint"
    }

    INDUSTRY_SECTORS = {
        "banking",
        "finance",
        "financial services",
        "manufacturing",
        "technology",
        "construction",
        "healthcare",
        "education",
        "government",
        "telecommunications",
        "energy",
        "retail",
        "transportation",
        "aerospace",
        "defense",
        "insurance",
        "automotive",
        "agriculture",
        "hospitality",
        "media",
    }

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

    def mask_non_text_artifacts(self, text: str) -> str:
        """
        Mask Markdown image destinations, data URIs, and URLs
        while preserving the original character positions.
        """
        masked = list(text)
        patterns = [
            # Markdown image destinations (with whitespace tolerance inside parentheses)
            r'!\[[^\]]*\]\s*\(\s*[^)]+\)',
            # Markdown link destinations (with whitespace tolerance inside parentheses)
            r'\[[^\]]+\]\s*\(\s*[^)]+\)',
            r'\]\s*\(\s*[^)]+\)',
            # Base64 data URIs
            r'data:image/[^,\s)]+;base64,[A-Za-z0-9+/=\s]+',
            # Ordinary URLs
            r'https?://[^\s<>"\']+',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                for i in range(match.start(), match.end()):
                    if masked[i] not in "\r\n":
                        masked[i] = " "
        return "".join(masked)

    def predict_sentence(self, sentence: PreprocessedSentence) -> List[Dict[str, Any]]:
        """Predict entity categories for tokens in a preprocessed sentence."""
        if not self.is_trained or not self.pipeline:
            return []

        # Mask non-text artifacts before ML tokenization
        masked_text = self.mask_non_text_artifacts(sentence.text)
        # Tokenize while preserving original character offsets
        tokens = self.tokenize_with_spans(masked_text)
        if not tokens:
            return []

        features_list = [self.extract_token_features(tokens, i) for i in range(len(tokens))]
        predictions = self.pipeline.predict(features_list)
        probs = self.pipeline.predict_proba(features_list)
        classes = list(self.pipeline.classes_)

        results: List[Dict[str, Any]] = []
        for (tok_text, s_start, s_end), pred_label, prob_dist in zip(tokens, predictions, probs):
            if pred_label != "O":
                tok_lower = tok_text.lower()

                # Filter out malformed artifacts (MD5 hashes, divider lines, URL slugs)
                if self.is_non_entity_artifact(tok_text):
                    continue
                
                # Filter out generic noun fragments that should not stand alone
                if tok_lower in self.GENERIC_NON_ENTITIES:
                    continue

                # Named entities (ThreatActor, Organization, Location) must follow proper noun capitalization
                if pred_label in ("ThreatActor", "Organization", "Location"):
                    has_digit_and_letter = (
                        any(c.isdigit() for c in tok_text)
                        and any(c.isalpha() for c in tok_text)
                    )
                    is_proper_noun = (
                        tok_text.istitle()
                        or tok_text.isupper()
                        or has_digit_and_letter
                        or "-" in tok_text
                    )
                    is_known_sector = (
                        pred_label == "Organization"
                        and tok_lower in self.INDUSTRY_SECTORS
                    )
                    if not (is_proper_noun or is_known_sector):
                        continue
                
                # Prevent security vendors from being misclassified as threat actors or malware
                if tok_lower in self.SECURITY_VENDORS and pred_label in ("ThreatActor", "Malware"):
                    continue

                # Enforce syntactic constraints for CVE and Technique tokens
                if pred_label == "CVE" and not tok_lower.startswith("cve-"):
                    continue
                if pred_label == "Technique" and not re.match(r"^t\d{4}", tok_lower):
                    continue

                conf = float(prob_dist[classes.index(pred_label)])
                if conf >= 0.65:
                    results.append({
                        "text": tok_text,
                        "category": pred_label,
                        "canonical_name": tok_text,
                        "canonical_id": f"ML_{pred_label}_{tok_text}",
                        "start_char": s_start,
                        "end_char": s_end,
                        "confidence": round(conf, 4),
                        "model": "CyberMLNER",
                    })

        return results
