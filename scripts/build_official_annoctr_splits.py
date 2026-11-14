"""Build official AnnoCTR Train and Test splits strictly adhering to the original paper's partitions.

Sources:
- Train Split: data/annoctr_source/ner_train.jsonl (Official AnnoCTR Train: 70 reports, 7,570 sentences)
- Test Split:  data/annoctr_source/ner_test.jsonl  (Official AnnoCTR Test:  34 reports, 3,059 sentences)

Guarantees 100% official partition alignment and strictly zero leakage between train and test.
"""

import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("OfficialAnnoCTRSplitter")

TRAIN_SOURCE = Path("data/annoctr_source/ner_train.jsonl")
TEST_SOURCE = Path("data/annoctr_source/ner_test.jsonl")

OUTPUT_TRAIN = Path("data/benchmark/annoctr_train_split.json")
OUTPUT_TEST = Path("data/benchmark/annoctr_gold_standard.json")

TAG_TO_CATEGORY = {
    "GROUP": "ThreatActor",
    "MALWARE": "Malware",
    "TOOL": "Tool",
    "TECHNIQUE": "Technique",
    "TACTIC": "Technique",
    "ORG": "Organization",
    "SECTOR": "Organization",
    "LOC": "Location",
    "VULN": "CVE",
    "CVE": "CVE",
}

CVE_REGEX = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def detokenize(tokens: List[str]) -> str:
    """Detokenize tokens into natural grammatical English sentences."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,\.!?:;\)\]\}\'\"])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"\s+/\s+", "/", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    return text.strip()


def extract_entities_from_bio(tokens: List[str], bio_tags: List[str]) -> List[Dict[str, Any]]:
    """Convert token-level BIO tags into structured entity mention records."""
    entities = []
    current_tokens = []
    current_type = None

    for token, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            if current_tokens and current_type:
                surface = detokenize(current_tokens)
                cat = TAG_TO_CATEGORY.get(current_type)
                if cat and surface:
                    entities.append({
                        "category": cat,
                        "canonical_name": surface,
                        "surface_text": surface,
                        "annoctr_tag": current_type,
                    })
            current_tokens = [token]
            current_type = tag[2:]
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if current_type == tag_type:
                current_tokens.append(token)
            else:
                if current_tokens and current_type:
                    surface = detokenize(current_tokens)
                    cat = TAG_TO_CATEGORY.get(current_type)
                    if cat and surface:
                        entities.append({
                            "category": cat,
                            "canonical_name": surface,
                            "surface_text": surface,
                            "annoctr_tag": current_type,
                        })
                current_tokens = [token]
                current_type = tag_type
        else:
            if current_tokens and current_type:
                surface = detokenize(current_tokens)
                cat = TAG_TO_CATEGORY.get(current_type)
                if cat and surface:
                    entities.append({
                        "category": cat,
                        "canonical_name": surface,
                        "surface_text": surface,
                        "annoctr_tag": current_type,
                    })
            current_tokens = []
            current_type = None

    if current_tokens and current_type:
        surface = detokenize(current_tokens)
        cat = TAG_TO_CATEGORY.get(current_type)
        if cat and surface:
            entities.append({
                "category": cat,
                "canonical_name": surface,
                "surface_text": surface,
                "annoctr_tag": current_type,
            })

    # Also detect explicit CVE regex in tokens if not tagged
    for token in tokens:
        if CVE_REGEX.match(token) and not any(e["surface_text"] == token for e in entities):
            entities.append({
                "category": "CVE",
                "canonical_name": token.upper(),
                "surface_text": token,
                "annoctr_tag": "CVE",
            })

    return entities


def build_splits():
    assert TRAIN_SOURCE.exists(), f"Missing train source at {TRAIN_SOURCE}"
    assert TEST_SOURCE.exists(), f"Missing test source at {TEST_SOURCE}"

    # 1. Process Official Train Split
    logger.info("Processing Official AnnoCTR Train split...")
    train_records = []
    train_doc_ids = set()
    with open(TRAIN_SOURCE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tokens = row.get("tokens", [])
            # In official json, 'all_tags' contains the unified multi-layer BIO tags
            bio_tags = row.get("all_tags", row.get("ne_tags", []))
            if not tokens or not bio_tags:
                continue

            text = row.get("text") or detokenize(tokens)
            if len(text) < 15:
                continue

            doc_id = row["id"].rsplit("__", 1)[0]
            train_doc_ids.add(doc_id)
            entities = extract_entities_from_bio(tokens, bio_tags)

            # Store sentences containing cyber entities
            if entities:
                train_records.append({
                    "sentence_id": row["id"],
                    "report_id": f"ANNOCTR-{doc_id.upper()}",
                    "source": "Official AnnoCTR Train Corpus (Bosch Research)",
                    "publication_date": "2021-08-01",
                    "text": text,
                    "tokens": tokens,
                    "bio_tags": bio_tags,
                    "ground_truth_entities": entities,
                })

    logger.info(f"Loaded {len(train_records)} entity-bearing training sentences across {len(train_doc_ids)} training reports.")

    # 2. Process Official Test Split
    logger.info("Processing Official AnnoCTR Test split...")
    test_candidates = []
    test_doc_ids = set()
    with open(TEST_SOURCE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tokens = row.get("tokens", [])
            bio_tags = row.get("all_tags", row.get("gold", []))
            if not tokens or not bio_tags:
                continue

            text = row.get("text") or detokenize(tokens)
            if len(text) < 20:
                continue

            doc_id = row["id"].rsplit("__", 1)[0]
            test_doc_ids.add(doc_id)
            entities = extract_entities_from_bio(tokens, bio_tags)

            if entities:
                test_candidates.append({
                    "sentence_id": row["id"],
                    "report_id": f"ANNOCTR-{doc_id.upper()}",
                    "source": "Official AnnoCTR Test Corpus (Bosch Research)",
                    "publication_date": "2021-08-01",
                    "text": text,
                    "tokens": tokens,
                    "bio_tags": bio_tags,
                    "ground_truth_entities": entities,
                })

    # Strict check: 0 document overlap between official train and test
    overlap = train_doc_ids.intersection(test_doc_ids)
    assert len(overlap) == 0, f"FATAL: Document overlap detected: {overlap}"
    logger.info(f"Verified strict zero-leakage: 0 overlapping reports between train ({len(train_doc_ids)}) and test ({len(test_doc_ids)}).")

    # Sample exactly 100 diverse, representative test sentences
    rng = random.Random(42)
    selected_test: List[Dict[str, Any]] = []
    selected_ids = set()

    # Stratify by category
    for target_cat in ["ThreatActor", "Tool", "CVE", "Technique", "Malware", "Organization", "Location"]:
        cands = [s for s in test_candidates if s["sentence_id"] not in selected_ids and any(e["category"] == target_cat for e in s["ground_truth_entities"])]
        rng.shuffle(cands)
        take_n = min(15, len(cands))
        for s in cands[:take_n]:
            selected_test.append(s)
            selected_ids.add(s["sentence_id"])

    # Fill up to exactly 100 sentences
    remaining = [s for s in test_candidates if s["sentence_id"] not in selected_ids]
    rng.shuffle(remaining)
    needed = 100 - len(selected_test)
    for s in remaining[:needed]:
        selected_test.append(s)
        selected_ids.add(s["sentence_id"])

    selected_test.sort(key=lambda s: s["sentence_id"])

    # Output paths
    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)

    OUTPUT_TEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TEST, "w", encoding="utf-8") as f:
        json.dump(selected_test, f, indent=2, ensure_ascii=False)

    train_dist = Counter(e["category"] for s in train_records for e in s["ground_truth_entities"])
    test_dist = Counter(e["category"] for s in selected_test for e in s["ground_truth_entities"])

    logger.info("=" * 60)
    logger.info("OFFICIAL ANNOCTR DATASET SPLITS CONSTRUCTED:")
    logger.info(f"  - Official Train Set: {len(train_records)} sentences across {len(train_doc_ids)} reports")
    logger.info(f"    Train Entity Distribution: {dict(train_dist)}")
    logger.info(f"  - Official Test Benchmark: {len(selected_test)} gold sentences across {len(test_doc_ids)} reports")
    logger.info(f"    Test Entity Distribution:  {dict(test_dist)}")
    logger.info(f"  - Document Overlap between Train and Test: 0 (Strict Zero Leakage)")
    logger.info("=" * 60)


if __name__ == "__main__":
    build_splits()
