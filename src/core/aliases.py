"""MITRE-registry-based alias resolver (replaces the old hand-written alias table).

Aliases come from the shipped MITRE ATT&CK registry (threat actors, malware,
tools). A small reviewed override file adds common short names, sector
synonyms and "distinct entity" exclusions. A malware family is never treated as
an alias of a group, so ``BlackEnergy`` does not satisfy a ``Sandworm`` gate.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

GENERIC_STOPS = frozenset(
    {
        "group", "groups", "team", "teams", "unit", "units", "threat", "threats",
        "actor", "actors", "campaign", "campaigns", "operation", "operations",
        "malware", "tool", "tools", "trojan", "loader", "stealer", "backdoor",
        "ransomware", "virus", "worm", "apt", "corp", "corporation", "inc", "ltd",
        "organization", "organizations", "org", "orgs", "agency", "agencies",
        "company", "companies", "sector", "sectors", "industry", "industries",
        "government", "governments", "target", "targets", "targeted", "targeting",
        "associated", "related", "the", "and", "for", "with", "from", "snake", "beacon",
    }
)

_CATEGORY_BY_SECTION = {"threat_actors": "ThreatActor", "malware": "Malware", "tools": "Tool"}
_PREFIX_RE = re.compile(
    r"^(?:ThreatActor|Malware|Tool|Organization|Location|Technique|Vulnerability|CVE)::"
)
_TRAILING_GENERIC_RE = re.compile(r"\s+(?:group|team|apt)$", re.I)


@dataclass(frozen=True)
class ResolvedEntity:
    query: str
    canonical_name: str
    category: Optional[str]
    mitre_id: Optional[str]
    aliases: Tuple[str, ...]
    terms: FrozenSet[str]
    in_registry: bool
    exclusions: Tuple[str, ...] = field(default_factory=tuple)
    ambiguous: bool = False

    def __hash__(self) -> int:  # frozen dataclass with tuple/frozenset fields
        return hash((self.canonical_name, self.mitre_id, self.category))


def _clean_alias(alias: str) -> Optional[str]:
    """Drop registry artefacts such as 'BlackEnergy (Group' (unbalanced parentheses)."""
    a = (alias or "").strip()
    if not a:
        return None
    if a.count("(") != a.count(")") or "(" in a:
        return None
    return a


def _variants(term: str) -> Iterable[str]:
    t = term.lower().strip()
    yield t
    for ch in (" ", "-", "_"):
        if ch in t:
            yield t.replace(ch, "")


@functools.lru_cache(maxsize=4096)
def _terms_pattern(terms: FrozenSet[str]) -> Optional["re.Pattern[str]"]:
    """One whole-word alternation per entity, compiled once (longest terms first)."""
    if not terms:
        return None
    alt = "|".join(re.escape(t) for t in sorted(terms, key=lambda t: (-len(t), t)))
    return re.compile(r"(?<![\w])(?:" + alt + r")(?![\w])")


def normalize_name(name: str) -> str:
    return _PREFIX_RE.sub("", (name or "").strip()).strip().lower()


class AliasResolver:
    """Resolve any entity string to canonical identity plus a set of safe match terms."""

    def __init__(self, registry_path: Optional[Path] = None, overrides_path: Optional[Path] = None):
        self._by_key: Dict[str, List[dict]] = {}
        self._entities: Dict[str, dict] = {}
        self._sectors: Dict[str, List[str]] = {}
        self._exclusions: Dict[str, List[str]] = {}
        self._removed: set[str] = set()
        self.registry_size = 0
        if registry_path and Path(registry_path).exists():
            self._load_registry(Path(registry_path))
        if overrides_path and Path(overrides_path).exists():
            self._load_overrides(Path(overrides_path))

    # ------------------------------------------------------------------ load
    def _load_registry(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for section, category in _CATEGORY_BY_SECTION.items():
            for key, rec in (data.get(section) or {}).items():
                name = rec.get("name") or key
                ent = {
                    "name": name,
                    "category": category,
                    "mitre_id": rec.get("mitre_id"),
                    "aliases": [],
                }
                seen = {name.lower()}
                for al in rec.get("aliases", []):
                    cleaned = _clean_alias(al)
                    if cleaned and cleaned.lower() not in seen:
                        seen.add(cleaned.lower())
                        ent["aliases"].append(cleaned)
                ent_key = f"{category}::{rec.get('mitre_id') or name}"
                self._entities[ent_key] = ent
                self.registry_size += 1
                for form in {name, *ent["aliases"]}:
                    self._by_key.setdefault(form.lower(), []).append(ent)

    def _load_overrides(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for short, target in (data.get("short_names") or {}).items():
            ent = self._find_by_id_or_name(target)
            if ent is not None:
                self._by_key.setdefault(short.lower(), []).insert(0, ent)
                if short.lower() not in {a.lower() for a in ent["aliases"]} and short.lower() != ent["name"].lower():
                    ent["aliases"].append(short)
        for sector, syns in (data.get("sectors") or {}).items():
            self._sectors[sector.lower()] = [s.lower() for s in syns]
        for name, patterns in (data.get("exclusions") or {}).items():
            self._exclusions[name.lower()] = list(patterns)
        for form in data.get("remove_aliases", []):
            self._removed.add(form.lower())

    def _find_by_id_or_name(self, ref: str) -> Optional[dict]:
        for ent in self._entities.values():
            if ent["mitre_id"] == ref or ent["name"].lower() == ref.lower():
                return ent
        return None

    # --------------------------------------------------------------- resolve
    def resolve(self, name: str) -> ResolvedEntity:
        raw = (name or "").strip()
        key = normalize_name(raw)
        candidates = self._by_key.get(key, [])
        if not candidates:
            stripped = _TRAILING_GENERIC_RE.sub("", key)
            if stripped and stripped != key:
                candidates = self._by_key.get(stripped, [])
        exclusions = tuple(self._exclusions.get(key, ()))
        if candidates:
            exact = [c for c in candidates if c["name"].lower() == key]
            ent = (exact or candidates)[0]
            distinct_ids = {c["mitre_id"] for c in candidates}
            forms = [ent["name"], *ent["aliases"]]
            terms = self._terms(forms)
            return ResolvedEntity(
                query=raw,
                canonical_name=ent["name"],
                category=ent["category"],
                mitre_id=ent["mitre_id"],
                aliases=tuple(ent["aliases"]),
                terms=terms,
                in_registry=True,
                exclusions=exclusions,
                ambiguous=len(distinct_ids) > 1 and not exact,
            )
        # Not in the registry: the literal name plus reviewed sector synonyms only.
        forms = [raw, *self._sectors.get(key, [])]
        return ResolvedEntity(
            query=raw,
            canonical_name=_PREFIX_RE.sub("", raw).strip() or raw,
            category=None,
            mitre_id=None,
            aliases=tuple(self._sectors.get(key, ())),
            terms=self._terms(forms, allow_generic=True),
            in_registry=False,
            exclusions=exclusions,
        )

    def _terms(self, forms: Iterable[str], allow_generic: bool = False) -> FrozenSet[str]:
        out: set[str] = set()
        for form in forms:
            base = normalize_name(form)
            if not base or base in self._removed:
                continue
            for v in _variants(base):
                if len(v) >= 2 and (allow_generic or v not in GENERIC_STOPS):
                    out.add(v)
        return frozenset(out)

    # ----------------------------------------------------------------- match
    def present(self, text: str, entity: ResolvedEntity) -> bool:
        """Whole-word match of any term in ``text`` after removing excluded look-alikes."""
        low = (text or "").lower()
        for pat in entity.exclusions:
            low = re.sub(pat, " ", low)
        pat = _terms_pattern(entity.terms)
        return bool(pat and pat.search(low))

    def lookup(self, name: str) -> Optional[ResolvedEntity]:
        ent = self.resolve(name)
        return ent if ent.in_registry else None
