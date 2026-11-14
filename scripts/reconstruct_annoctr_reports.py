"""Reconstruct authentic CTI reports directly from original AnnoCTR corpus token sequences.

Dataset Source:
- Bosch Research AnnoCTR Corpus (LREC-COLING 2024)
- Repository: https://github.com/boschresearch/anno-ctr-lrec-coling-2024
- Documents: 104 authentic threat intelligence reports (34 test + 70 train)

Provenance Note:
- All texts are reconstructed from AnnoCTR token sequences via detokenization.
- URLs are explicitly set to None (no fabricated external URLs).
- Documents overlap with the AnnoCTR dataset documents as documented.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ReconstructAnnoCTR")

SOURCE_FILES = [
    Path("data/annoctr_source/ner_test.jsonl"),
    Path("data/annoctr_source/ner_train.jsonl"),
]
OUTPUT_PATH = Path("data/raw/cti_reports.json")


def detokenize(tokens: List[str]) -> str:
    """Cleanly join tokens into a readable grammatical sentence."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,\.!?:;\)\]\}\'\"])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"\s+/\s+", "/", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    return text.strip()


def process_slug_to_title(slug: str) -> str:
    """Convert a hyphenated slug into a clean professional title."""
    words = slug.split("-")
    clean_words = []
    acronyms = {"cve", "apt", "ta", "rat", "dll", "c2", "dns", "doh", "mfa", "psa", "unc", "fin"}
    for w in words:
        if w.lower() in acronyms:
            clean_words.append(w.upper())
        else:
            clean_words.append(w.capitalize())
    return " ".join(clean_words)


def build_reports() -> List[Dict[str, Any]]:
    """Build structured CTI reports from AnnoCTR corpus."""
    docs: Dict[str, List[Dict[str, Any]]] = {}

    for source_file in SOURCE_FILES:
        if not source_file.exists():
            logger.error(f"Source file not found: {source_file}")
            continue
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                doc_id = item["id"].split("__")[0]
                docs.setdefault(doc_id, []).append(item)

    logger.info(f"Loaded {len(docs)} documents across source files.")

    reports = []
    publisher_map = {
        "proofpoint": "Proofpoint",
        "zscaler": "Zscaler",
        "intel471": "Intel 471",
        "lab52": "Lab52",
        "quoteintelligence": "QuoIntelligence",
        "quointelligence": "QuoIntelligence",
    }

    for doc_id, sentences_data in docs.items():
        parts = doc_id.split("_", 2)
        publisher_raw = parts[0].lower()
        date_str = parts[1] if len(parts) > 1 else "2021-01-01"
        slug = parts[2] if len(parts) > 2 else doc_id

        publisher = publisher_map.get(publisher_raw, parts[0].capitalize())
        title = process_slug_to_title(slug)

        # Detokenize sentences and form readable paragraphs (~4 sentences each)
        cleaned_sentences = [detokenize(s["tokens"]) for s in sentences_data if s.get("tokens")]
        paragraphs = []
        chunk_size = 4
        for i in range(0, len(cleaned_sentences), chunk_size):
            para = " ".join(cleaned_sentences[i : i + chunk_size])
            if para.strip():
                paragraphs.append(para.strip())

        full_content = "\n\n".join(paragraphs)

        report_entry = {
            "report_id": f"ANNOCTR-{publisher_raw.upper()}-{date_str}-{slug[:25].upper()}",
            "title": title,
            "source": publisher,
            "publication_date": date_str,
            "tlp": "CLEAR",
            "author": f"{publisher} Threat Intelligence Team",
            "url": None,  # Transparently null; avoid fabricated URLs
            "content": full_content,
            "extra_metadata": {
                "annoctr_doc_id": doc_id,
                "total_annotated_sentences": len(sentences_data),
                "corpus_source": "Bosch Research AnnoCTR Corpus (LREC-COLING 2024)",
                "corpus_repo": "https://github.com/boschresearch/anno-ctr-lrec-coling-2024",
                "text_nature": "Reconstructed AnnoCTR report texts from corpus token sequences (not downloaded from original publisher websites)",
                "notes": "Reconstructed from token sequences of the AnnoCTR dataset. URLs are set to null because original external publisher links are not verified. Overlaps with AnnoCTR dataset documents.",
            },
        }
        reports.append(report_entry)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    total_words = sum(len(r["content"].split()) for r in reports)
    logger.info(f"Successfully generated {len(reports)} authentic AnnoCTR reports into {OUTPUT_PATH}")
    logger.info(f"Total word count: {total_words:,} words across {len(reports)} reports.")
    return reports


if __name__ == "__main__":
    build_reports()
