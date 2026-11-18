"""Extract balanced, stratified original AnnoCTR human-annotated sentences from ner_predictions_test.jsonl.
Preserves original token sequences, gold BIO entity spans, and document provenance across all categories.
"""

import json
import re
import random
import os
from pathlib import Path
from typing import List, Dict, Any

INPUT_PATH = Path(os.environ.get("ANNOCTR_TEST_JSONL", "data/annoctr_source/ner_test.jsonl"))
OUTPUT_PATH = Path("data/benchmark/annoctr_gold_standard.json")

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

    # Also extract explicit CVEs if present in tokens
    sentence_text = detokenize(tokens)
    for m in CVE_REGEX.finditer(sentence_text):
        cve_id = m.group(0).upper()
        if not any(e["canonical_name"] == cve_id for e in entities):
            entities.append({
                "category": "CVE",
                "canonical_name": cve_id,
                "surface_text": cve_id,
                "annoctr_tag": "CVE",
            })

    return entities

def build_gold_benchmark(total_target: int = 100):
    all_sentences = []
    by_category: Dict[str, List[Dict[str, Any]]] = {}

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = item.get("tokens", [])
            bio_tags = item.get("gold", [])
            
            entities = extract_entities_from_bio(tokens, bio_tags)
            if not entities:
                continue

            doc_id = item["id"].split("__")[0]
            parts = doc_id.split("_", 2)
            publisher_raw = parts[0]
            date_str = parts[1] if len(parts) > 1 else "2021-01-01"

            publisher_map = {
                "proofpoint": "Proofpoint",
                "zscaler": "Zscaler",
                "intel471": "Intel 471",
                "lab52": "Lab52",
                "quoteintelligence": "QuoteIntelligence",
            }
            publisher = publisher_map.get(publisher_raw.lower(), publisher_raw.capitalize())
            sentence_text = detokenize(tokens)

            record = {
                "sentence_id": item["id"],
                "report_id": f"ANNOCTR-{publisher_raw.upper()}-{date_str}",
                "source": publisher,
                "publication_date": date_str,
                "text": sentence_text,
                "tokens": tokens,
                "bio_tags": bio_tags,
                "ground_truth_entities": entities,
            }
            all_sentences.append(record)

            for e in entities:
                by_category.setdefault(e["category"], []).append(record)

    # Stratified selection to ensure Tool, ThreatActor, CVE, Location, Malware, Organization, Technique
    random.seed(42)
    selected_dict = {}

    # Prioritize minority categories
    for cat in ["Tool", "CVE", "ThreatActor", "Location", "Malware", "Organization", "Technique"]:
        pool = by_category.get(cat, [])
        random.shuffle(pool)
        for s in pool:
            if s["sentence_id"] not in selected_dict:
                selected_dict[s["sentence_id"]] = s
                if len(selected_dict) >= total_target:
                    break
        if len(selected_dict) >= total_target:
            break

    selected = list(selected_dict.values())
    cat_counts = {}
    for s in selected:
        for e in s["ground_truth_entities"]:
            cat_counts[e["category"]] = cat_counts.get(e["category"], 0) + 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(selected)} original AnnoCTR annotated sentences to {OUTPUT_PATH}")
    print(f"Category breakdown across selected benchmark: {cat_counts}")

if __name__ == "__main__":
    build_gold_benchmark(100)
