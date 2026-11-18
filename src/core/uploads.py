"""Upload validation and text extraction (PDF/TXT/MD/HTML) with typed rejections (R6)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List, Tuple

from src.core.errors import NoExtractableTextError, UploadRejectedError

ALLOWED_EXT = (".pdf", ".txt", ".md", ".markdown", ".html", ".htm")


class _TextOnly(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextOnly()
    parser.feed(html)
    text = "".join(parser.parts)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def decode_text_bytes(data: bytes) -> Tuple[str, List[str]]:
    """Decode with fallbacks; reject binary content disguised as text."""
    warnings: List[str] = []
    if b"\x00" in data[:8192]:
        raise UploadRejectedError("File looks binary (NUL bytes) and is not a text report")
    for enc in ("utf-8-sig", "utf-16"):
        try:
            if enc == "utf-16" and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return data.decode(enc), warnings
        except UnicodeDecodeError:
            continue
    warnings.append("Not valid UTF-8; decoded with cp1252 fallback")
    return data.decode("cp1252", errors="replace"), warnings


def _control_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for c in text if (ord(c) < 32 and c not in "\n\r\t") or c == "�")
    return bad / len(text)


def validate_report_text(text: str, max_bytes: int) -> str:
    """Common gate for text that arrives directly (MCP) or after extraction."""
    if not isinstance(text, str):
        raise UploadRejectedError("Report text must be a string")
    if len(text.encode("utf-8", errors="ignore")) > max_bytes:
        raise UploadRejectedError(f"Report text exceeds the {max_bytes // (1024 * 1024)} MB limit")
    if "\x00" in text:
        raise UploadRejectedError("Report text contains NUL bytes; it does not look like text")
    if _control_ratio(text) > 0.05:
        raise UploadRejectedError("Report text is mostly control or replacement characters")
    cleaned = text.strip()
    if len(re.findall(r"[A-Za-z]{2,}", cleaned)) < 5:
        raise NoExtractableTextError("No readable text found in the report")
    return cleaned


def extract_text_from_upload(
    data: bytes, filename: str, max_bytes: int, max_pages: int
) -> Tuple[str, List[str]]:
    """Return (text, warnings) or raise UploadRejectedError / NoExtractableTextError."""
    if not data:
        raise UploadRejectedError("File is empty")
    if len(data) > max_bytes:
        raise UploadRejectedError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
    name = (filename or "").lower()
    ext = next((e for e in ALLOWED_EXT if name.endswith(e)), None)
    if ext is None:
        raise UploadRejectedError(f"Unsupported file type: {filename!r}")

    warnings: List[str] = []
    if ext == ".pdf":
        if not data.lstrip().startswith(b"%PDF"):
            raise UploadRejectedError("File has a .pdf extension but is not a PDF")
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise UploadRejectedError(f"PDF could not be opened: {exc}") from exc
        try:
            if doc.needs_pass or doc.is_encrypted:
                raise UploadRejectedError("PDF is encrypted; remove the password and retry")
            if doc.page_count > max_pages:
                raise UploadRejectedError(f"PDF has {doc.page_count} pages; the limit is {max_pages}")
            pages = [p.get_text("text") for p in doc]
        finally:
            doc.close()
        text = "\n\n".join(pages)
        if len(re.findall(r"[A-Za-z]{2,}", text)) < 5:
            raise NoExtractableTextError("PDF has no extractable text (scanned?); run OCR first")
        return validate_report_text(text, max_bytes), warnings

    text, warnings = decode_text_bytes(data)
    if ext in (".html", ".htm"):
        text = html_to_text(text)
    return validate_report_text(text, max_bytes), warnings
