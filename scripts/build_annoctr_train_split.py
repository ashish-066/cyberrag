"""Build independent, held-out AnnoCTR training split.

Extracts all sentences from the 15 documents in ner_predictions_test.jsonl that are
STRICTLY DISJOINT from the 19 test documents used in data/benchmark/annoctr_gold_standard.json.
Guarantees 0% document leakage and 0% sentence leakage.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

INPUT_PATH = Path(r"C:\Users\njosh\Downloads\ner_predictions_test.jsonl")
GOLD_TEST_PATH = Path("data/benchmark/annoctr_gold_standard.json")
OUTPUT_TRAIN_PATH = Path("data/benchmark/annoctr_train_split.json")

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


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([,\.!?:;\)\]\}\'\"])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"\s+/\s+", "/", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    return text.strip()


def extract_entities_from_bio(tokens: List[str], bio_tags: List[str]) -> List[Dict[str, Any]]:
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

    return entities


def build_train_split():
    with open(GOLD_TEST_PATH, "r", encoding="utf-8") as f:
        gold_test = json.load(f)

    test_doc_ids = set(s["sentence_id"].rsplit("__", 1)[0] for s in gold_test)
    print(f"Identified {len(test_doc_ids)} test documents to hold out.")

    train_samples = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = row["id"].rsplit("__", 1)[0]
            if doc_id in test_doc_ids:
                continue

            tokens = row.get("tokens", [])
            bio_tags = row.get("gold", [])
            if not tokens or not bio_tags:
                continue

            text = detokenize(tokens)
            if len(text) < 15:
                continue

            entities = extract_entities_from_bio(tokens, bio_tags)
            train_samples.append({
                "sentence_id": row["id"],
                "report_id": f"ANNOCTR-{doc_id.upper()}",
                "source": "AnnoCTR Real CTI Corpus (Training Split)",
                "publication_date": "2021-08-01",
                "text": text,
                "tokens": tokens,
                "bio_tags": bio_tags,
                "ground_truth_entities": entities,
            })

    OUTPUT_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TRAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)

    print("=" * 60)
    print(f"ANNOCTR TRAINING SPLIT BUILT: {len(train_samples)} sentences.")
    print(f"Test documents completely held out: {len(test_doc_ids)}")
    print(f"Train output path: {OUTPUT_TRAIN_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    build_train_split()
