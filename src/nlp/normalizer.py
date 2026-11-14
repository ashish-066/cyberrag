"""Entity Normalizer and Disambiguator for Cybersecurity Entities.

Features:
- Disambiguates similarly named entities (e.g. Cobalt Strike framework vs BEACON payload; Carbanak group vs malware)
- Preserves exact verbatim surface forms and character offsets
- Stores standardized canonical IDs (MITRE G/S/T codes, CVEs, ISO country codes)
- Records ambiguous candidate matches instead of silently guessing
"""

import re
from typing import Dict, Optional, Tuple, Any, List

from .gazetteer import (
    THREAT_ACTORS,
    MALWARE,
    TOOLS,
    KNOWN_CVES,
    TECHNIQUES,
    ORGANIZATIONS,
    LOCATIONS,
)


class EntityNormalizer:
    """Normalizes, links, and disambiguates extracted cybersecurity entities to canonical concepts."""

    def __init__(self):
        # Maps lowercase alias -> List of (category, canonical_name, metadata)
        self.alias_to_candidates: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        self._build_index()

    def _build_index(self):
        """Construct multi-candidate lookup tables for high-speed resolution and ambiguity detection."""
        # 1. Threat Actors
        for canon, data in THREAT_ACTORS.items():
            for alias in data["aliases"]:
                self._add_candidate(alias.lower(), "ThreatActor", canon, data)

        # 2. Malware
        for canon, data in MALWARE.items():
            for alias in data["aliases"]:
                self._add_candidate(alias.lower(), "Malware", canon, data)

        # 3. Tools
        for canon, data in TOOLS.items():
            for alias in data["aliases"]:
                self._add_candidate(alias.lower(), "Tool", canon, data)

        # 4. Known CVEs
        for canon, data in KNOWN_CVES.items():
            for alias in data["aliases"]:
                self._add_candidate(alias.lower(), "CVE", canon, data)

        # 5. MITRE Techniques
        for tech_id, data in TECHNIQUES.items():
            canon = f"{tech_id} - {data['name']}"
            for alias in data["aliases"]:
                self._add_candidate(alias.lower(), "Technique", canon, data)

        # 6. Organizations
        for canon, data in ORGANIZATIONS.items():
            for alias in data["aliases"]:
                self._add_candidate(alias.lower(), "Organization", canon, data)

        # 7. Locations
        for canon, data in LOCATIONS.items():
            for alias in data["aliases"]:
                self._add_candidate(alias.lower(), "Location", canon, data)

    def _add_candidate(self, key: str, cat: str, canon: str, data: Dict[str, Any]):
        """Add candidate record under lowercase key."""
        entry = (cat, canon, data)
        self.alias_to_candidates.setdefault(key, []).append(entry)

    def normalize_cve(self, raw_cve: str) -> Tuple[str, str]:
        """Normalize CVE identifier to canonical uppercase format."""
        match = re.search(r"CVE[-_ ](\d{4})[-_ ](\d{4,7})", raw_cve, re.IGNORECASE)
        if match:
            cve_id = f"CVE-{match.group(1)}-{match.group(2)}"
            return cve_id, cve_id
        return raw_cve.upper(), raw_cve.upper()

    def normalize_technique_id(self, raw_tech: str) -> Optional[Tuple[str, str]]:
        """Normalize technique mentions like T1566 to canonical format."""
        match = re.search(r"\b(T\d{4}(?:\.\d{3})?)\b", raw_tech, re.IGNORECASE)
        if match:
            t_id = match.group(1).upper()
            if t_id in TECHNIQUES:
                return f"{t_id} - {TECHNIQUES[t_id]['name']}", t_id
            return t_id, t_id
        return None

    def disambiguate_context(
        self,
        surface_text: str,
        candidates: List[Tuple[str, str, Dict[str, Any]]],
        surrounding_context: str = "",
    ) -> Tuple[Tuple[str, str, Dict[str, Any]], bool, List[str]]:
        """
        Disambiguate between multiple candidate interpretations based on contextual clues.
        Returns ((category, canonical_name, meta), is_ambiguous, candidate_descriptions).
        """
        candidate_descs = [f"{c[0]}::{c[1]}" for c in candidates]
        if len(candidates) == 1:
            return candidates[0], False, candidate_descs

        lower_ctx = surrounding_context.lower()
        lower_surf = surface_text.lower()

        # Rule 1: Carbanak disambiguation (Actor vs Malware)
        if "carbanak" in lower_surf:
            if any(w in lower_ctx for w in ("gang", "group", "actor", "threat", "adversary", "attributed to")):
                for c in candidates:
                    if c[0] == "ThreatActor":
                        return c, True, candidate_descs
            if any(w in lower_ctx for w in ("malware", "trojan", "backdoor", "deployed", "executed", "payload")):
                for c in candidates:
                    if c[0] == "Malware":
                        return c, True, candidate_descs

        # Rule 2: Cobalt Strike disambiguation (Tool Framework vs Malware Beacon)
        if "cobalt strike" in lower_surf:
            if "beacon" in lower_surf or "beacon" in lower_ctx or "payload" in lower_ctx:
                for c in candidates:
                    if c[0] == "Malware":
                        return c, True, candidate_descs
            for c in candidates:
                if c[0] == "Tool":
                    return c, True, candidate_descs

        # Default fallback: pick first candidate but flag as ambiguous
        return candidates[0], True, candidate_descs

    def normalize(
        self,
        surface_text: str,
        category: Optional[str] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Normalize and link entity surface text to canonical ID and name.
        Returns dict containing:
        - canonical_name
        - canonical_id
        - category
        - surface_text
        - is_ambiguous: bool
        - ambiguity_candidates: List[str]
        - metadata: Dict
        """
        cleaned = surface_text.strip()
        lower_key = cleaned.lower()

        # 1. Check exact alias matches in index
        if lower_key in self.alias_to_candidates:
            candidates = self.alias_to_candidates[lower_key]
            # Filter by category hint if provided
            if category:
                matched_cats = [c for c in candidates if c[0] == category]
                if matched_cats:
                    candidates = matched_cats

            chosen, is_ambig, all_candidates = self.disambiguate_context(cleaned, candidates, context)
            cat, canon, meta = chosen
            return {
                "canonical_name": canon,
                "canonical_id": meta.get("canonical_id", canon),
                "category": cat,
                "surface_text": cleaned,
                "is_ambiguous": is_ambig if len(candidates) > 1 else False,
                "ambiguity_candidates": all_candidates if len(candidates) > 1 else [],
                "metadata": meta,
            }

        # 2. Check CVE regex
        if category == "CVE" or re.search(r"CVE-\d{4}-\d{4,7}", cleaned, re.IGNORECASE):
            c_name, c_id = self.normalize_cve(cleaned)
            meta = KNOWN_CVES.get(c_name, {"canonical_id": c_id, "name": c_name})
            return {
                "canonical_name": c_name,
                "canonical_id": c_id,
                "category": "CVE",
                "surface_text": cleaned,
                "is_ambiguous": False,
                "ambiguity_candidates": [],
                "metadata": meta,
            }

        # 3. Check MITRE technique ID
        tech_norm = self.normalize_technique_id(cleaned)
        if tech_norm:
            c_name, c_id = tech_norm
            meta = TECHNIQUES.get(c_id, {"canonical_id": c_id, "name": c_name})
            return {
                "canonical_name": c_name,
                "canonical_id": c_id,
                "category": "Technique",
                "surface_text": cleaned,
                "is_ambiguous": False,
                "ambiguity_candidates": [],
                "metadata": meta,
            }

        # 4. Partial substring fallback
        matching_entries = []
        for alias, c_list in self.alias_to_candidates.items():
            if len(alias) >= 4 and (alias == lower_key or alias in lower_key):
                for c in c_list:
                    if not category or category == c[0]:
                        matching_entries.append(c)

        if matching_entries:
            chosen, is_ambig, all_candidates = self.disambiguate_context(cleaned, matching_entries, context)
            cat, canon, meta = chosen
            return {
                "canonical_name": canon,
                "canonical_id": meta.get("canonical_id", canon),
                "category": cat,
                "surface_text": cleaned,
                "is_ambiguous": True if len(matching_entries) > 1 else False,
                "ambiguity_candidates": all_candidates,
                "metadata": meta,
            }

        # Fallback to default
        final_cat = category or "Entity"
        return {
            "canonical_name": cleaned,
            "canonical_id": f"{final_cat}_{cleaned.replace(' ', '_')}",
            "category": final_cat,
            "surface_text": cleaned,
            "is_ambiguous": False,
            "ambiguity_candidates": [],
            "metadata": {},
        }
