"""Text cleaner and defanger for CTI reports."""

import re
import unicodedata
from typing import List, Tuple


class CTICleaner:
    """Cleans, normalizes, and refangs CTI report text while preserving semantic content."""

    # Common defanging patterns in threat intelligence
    DEFANG_PATTERNS: List[Tuple[re.Pattern, str]] = [
        # Bracketed colons and slashes in schemes
        (re.compile(r"hxxp(s?)(?:\[:\]|:)(?:\[/\]|/){2}", re.IGNORECASE), lambda m: f"http{m.group(1).lower()}://"),
        (re.compile(r"http(s?)(?:\[:\])(?:\[/\]|/){2}", re.IGNORECASE), lambda m: f"http{m.group(1).lower()}://"),
        (re.compile(r"\[(?:https?|hxxps?)\]://", re.IGNORECASE), "https://"),
        (re.compile(r"hxxps?://", re.IGNORECASE), lambda m: "https://" if "s" in m.group(0).lower() else "http://"),
        (re.compile(r"hXXp[sS]?://"), lambda m: "https://" if "s" in m.group(0).lower() else "http://"),
        # Bracketed dots in domains/IPs
        (re.compile(r"\[\.\]"), "."),
        (re.compile(r"\(\.\)"), "."),
        (re.compile(r"\{\.\}"), "."),
        (re.compile(r"\[dot\]", re.IGNORECASE), "."),
        # Bracketed colons
        (re.compile(r"\[:\]"), ":"),
        (re.compile(r"\(:/\)"), "://"),
        # Bracketed @ in emails
        (re.compile(r"\[@\]"), "@"),
        (re.compile(r"\[at\]", re.IGNORECASE), "@"),
    ]

    # Standard unicode replacements for clean ASCII/UTF-8
    UNICODE_REPLACEMENTS = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\ufeff": "",
    }

    # Boilerplate patterns to remove or clean
    BOILERPLATE_PATTERNS = [
        re.compile(r"^\s*(?:TLP:\w+|TRAFFIC LIGHT PROTOCOL.*)$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*DISCLAIMER:.*$", re.IGNORECASE | re.MULTILINE),
    ]

    def __init__(self, refang: bool = True, strip_boilerplate: bool = True):
        self.refang = refang
        self.strip_boilerplate = strip_boilerplate

    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode quotes, dashes, and invisible characters."""
        for orig, rep in self.UNICODE_REPLACEMENTS.items():
            text = text.replace(orig, rep)
        # Normalize NFKC
        text = unicodedata.normalize("NFKC", text)
        return text

    def refang_text(self, text: str) -> str:
        """Refang obfuscated CTI indicators like hxxp:// and evil[.]com."""
        for pattern, replacement in self.DEFANG_PATTERNS:
            if callable(replacement):
                text = pattern.sub(replacement, text)
            else:
                text = pattern.sub(replacement, text)
        return text

    def clean(self, text: str) -> str:
        """Run full cleaning pipeline on text."""
        if not text:
            return ""

        # Normalize unicode
        text = self.normalize_unicode(text)

        # Refang indicators
        if self.refang:
            text = self.refang_text(text)

        # Remove boilerplate if requested
        if self.strip_boilerplate:
            for pattern in self.BOILERPLATE_PATTERNS:
                text = pattern.sub("", text)

        # Standardize line endings and whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines).strip()

        return text
