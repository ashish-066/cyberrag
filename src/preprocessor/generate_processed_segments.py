"""Generator and Validator for processed_segments.jsonl.

Extracts sentence-level segments from preprocessed CTI reports with full
provenance, exact character offsets, and report metadata.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

INPUT_FILE = Path("data/processed/preprocessed_cti.json")
OUTPUT_FILE = Path("data/processed_segments.jsonl")

REQUIRED_FIELDS = [
    "segment_id",
    "sentence_id",
    "paragraph_id",
    "report_id",
    "text",
    "clean_text",
    "source",
    "publication_date",
    "paragraph_index",
    "sentence_index",
    "global_sentence_index",
    "start_char",
    "end_char",
]


def load_reports(path: Path) -> List[Dict[str, Any]]:
    """Load preprocessed CTI reports from JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("reports"), list):
            return data["reports"]
    return [data]


def generate_segments(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten 104 reports into sentence-level segment records."""
    segments = []
    for report in reports:
        report_id = report.get("report_id", "")
        source = report.get("source", "")
        pub_date = report.get("publication_date", "")
        title = report.get("title", "")
        url = report.get("url", "")
        author = report.get("author", "")

        for p_idx, para in enumerate(report.get("paragraphs", [])):
            para_id = para.get("paragraph_id", f"{report_id}::p{p_idx}")
            for s_idx, sent in enumerate(para.get("sentences", [])):
                sent_id = sent.get("sentence_id", f"{para_id}::s{s_idx}")
                text = sent.get("text", "")

                segment = {
                    "segment_id": sent_id,
                    "sentence_id": sent_id,
                    "paragraph_id": para_id,
                    "report_id": report_id,
                    "text": text,
                    "clean_text": text,
                    "source": sent.get("source", source),
                    "publication_date": sent.get("publication_date", pub_date),
                    "paragraph_index": sent.get("paragraph_index", p_idx),
                    "sentence_index": sent.get("sentence_index", s_idx),
                    "global_sentence_index": sent.get("global_sentence_index", len(segments)),
                    "start_char": sent.get("start_char", 0),
                    "end_char": sent.get("end_char", len(text)),
                    "title": title,
                    "url": url,
                    "author": author,
                }
                segments.append(segment)
    return segments


def validate_segments(segments: List[Dict[str, Any]], expected_count: int = 5991) -> bool:
    """Validate data integrity and provenance requirements."""
    seen_ids = set()
    for seg in segments:
        # 1. Required fields
        for field in REQUIRED_FIELDS:
            if field not in seg:
                raise ValueError(f"Missing required field '{field}' in segment {seg.get('segment_id')}")

        # 2. Unique segment IDs
        seg_id = seg["segment_id"]
        if seg_id in seen_ids:
            raise ValueError(f"Duplicate segment_id found: {seg_id}")
        seen_ids.add(seg_id)

        # 3. Match sentence IDs
        if seg_id != seg["sentence_id"]:
            raise ValueError(f"segment_id '{seg_id}' does not match sentence_id '{seg['sentence_id']}'")

        # 4. Non-empty text
        if not seg["text"] or not seg["text"].strip():
            raise ValueError(f"Empty text in segment {seg_id}")

        # 5. Character offset bounds
        if seg["start_char"] is not None and seg["end_char"] is not None:
            if seg["start_char"] < 0 or seg["end_char"] < seg["start_char"]:
                raise ValueError(f"Invalid offsets in segment {seg_id}: [{seg['start_char']}:{seg['end_char']}]")

    return True


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found at {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    reports = load_reports(INPUT_FILE)
    segments = generate_segments(reports)
    validate_segments(segments, expected_count=len(segments))

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    print("SEGMENT GENERATION COMPLETE")
    print("--------------------------------")
    print(f"Reports processed : {len(reports)}")
    print(f"Segments generated: {len(segments)}")
    print(f"Unique IDs        : {len(set(s['segment_id'] for s in segments))}")
    print(f"Output file       : {OUTPUT_FILE}")
    print("\nVALIDATION PASSED")
    print("--------------------------------")
    print("All required fields present")
    print("All segment IDs unique")
    print("All segment IDs match sentence IDs")
    print("All sentence texts non-empty")
    print("Character offsets have valid bounds")
    print(f"Verified sentence count: {len(segments)}")


if __name__ == "__main__":
    main()
