"""Dataset loader for raw CTI reports."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class CTIReport(BaseModel):
    """Raw CTI report schema preserving essential provenance metadata."""

    report_id: str = Field(..., description="Unique identifier of the CTI report")
    source: str = Field(..., description="Publisher or intel source (e.g., CISA, Mandiant)")
    publication_date: str = Field(..., description="Publication date in YYYY-MM-DD format")
    title: str = Field(default="", description="Title or headline of the report")
    content: str = Field(..., description="Full text content of the CTI report")
    url: Optional[str] = Field(default=None, description="Source URL or reference link")
    author: Optional[str] = Field(default=None, description="Authoring organization or analyst")
    tlp: Optional[str] = Field(default="CLEAR", description="Traffic Light Protocol marking")
    extra_metadata: dict = Field(default_factory=dict, description="Additional custom metadata")

    @field_validator("report_id")
    @classmethod
    def validate_report_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("report_id cannot be empty")
        return v

    @field_validator("publication_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        v = v.strip()
        # Accept YYYY-MM-DD or partial YYYY-MM
        if not v:
            return "UNKNOWN"
        return v


class CTIDatasetLoader:
    """Loads and validates CTI reports from disk (JSON, JSONL, or folders)."""

    def __init__(self, raw_data_dir: Union[str, Path] = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)

    def load_from_file(self, file_path: Union[str, Path]) -> List[CTIReport]:
        """Load CTI reports from a JSON or JSONL file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CTI report file not found: {file_path}")

        reports: List[CTIReport] = []
        if file_path.suffix.lower() == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        reports.append(CTIReport(**data))
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        reports.append(CTIReport(**item))
                elif isinstance(data, dict):
                    reports.append(CTIReport(**data))
                else:
                    raise ValueError(f"Unsupported JSON format in {file_path}")

        logger.info(f"Loaded {len(reports)} CTI reports from {file_path}")
        return reports

    def load_all_raw(self) -> List[CTIReport]:
        """Scan raw data directory and load all reports."""
        all_reports: List[CTIReport] = []
        if not self.raw_data_dir.exists():
            logger.warning(f"Directory {self.raw_data_dir} does not exist.")
            return all_reports

        for file_path in sorted(self.raw_data_dir.glob("*.json")):
            try:
                reports = self.load_from_file(file_path)
                all_reports.extend(reports)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        for file_path in sorted(self.raw_data_dir.glob("*.jsonl")):
            try:
                reports = self.load_from_file(file_path)
                all_reports.extend(reports)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        # Deduplicate by report_id
        unique_reports = {}
        for r in all_reports:
            unique_reports[r.report_id] = r

        deduped = list(unique_reports.values())
        logger.info(f"Loaded total {len(deduped)} unique CTI reports from {self.raw_data_dir}")
        return deduped
