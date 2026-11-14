"""Download the official AnnoCTR dataset splits from the official Bosch Research repository:
https://github.com/boschresearch/anno-ctr-lrec-coling-2024

Downloads:
- train.json -> data/annoctr_source/ner_train.jsonl (7,570 sentences, 70 reports)
- test.json  -> data/annoctr_source/ner_test.jsonl  (3,059 sentences, 34 reports)
- dev.json   -> data/annoctr_source/ner_dev.jsonl   (1,550 sentences, 20 reports)
"""

import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("AnnoCTRDownloader")

BASE_URL = "https://raw.githubusercontent.com/boschresearch/anno-ctr-lrec-coling-2024/main/AnnoCTR/ner_json/"
OUTPUT_DIR = Path("data/annoctr_source")


def download_splits():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {
        "train.json": "ner_train.jsonl",
        "test.json": "ner_test.jsonl",
        "dev.json": "ner_dev.jsonl",
    }

    for remote_name, local_name in splits.items():
        url = BASE_URL + remote_name
        dest = OUTPUT_DIR / local_name
        logger.info(f"Downloading {url} to {dest}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
            with open(dest, "wb") as f:
                f.write(content)

        line_count = sum(1 for _ in open(dest, "r", encoding="utf-8"))
        logger.info(f"Successfully saved {dest} ({line_count} sentences, {len(content) / 1024 / 1024:.2f} MB).")

    logger.info("=" * 60)
    logger.info("ALL OFFICIAL ANNOCTR DATASET SPLITS DOWNLOADED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    download_splits()
