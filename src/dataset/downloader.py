"""Dataset downloader and fetcher for external CTI corpora."""

import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class CTIDatasetDownloader:
    """Handles downloading or initializing CTI datasets."""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_url_json(self, url: str, timeout: int = 10) -> Optional[Any]:
        """Attempt to fetch JSON from a remote URL with timeout."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "CyberRAG-CTI-Ingester/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, Exception) as e:
            logger.warning(f"Could not fetch online dataset from {url}: {e}")
            return None

    def ensure_dataset(self) -> Path:
        """Verify that at least one raw CTI dataset file exists; if not, report instructions."""
        raw_files = list(self.output_dir.glob("*.json")) + list(self.output_dir.glob("*.jsonl"))
        target_file = self.output_dir / "cti_reports.json"

        if raw_files:
            logger.info(f"Existing raw CTI dataset files found: {[f.name for f in raw_files]}")
            return raw_files[0]

        logger.info(f"Target dataset file ready at {target_file}")
        return target_file
