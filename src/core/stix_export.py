"""STIX 2.1 bundle builder with validation.

Only SUPPORTED claims become Relationship objects. Every claim also gets a Note
with the verdict and external references pointing at the evidence ids, so a
consumer can see unsupported / conflicting results rather than silently losing
them. No confidence numbers are invented.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.core.aliases import AliasResolver
from src.core.errors import InternalError
from src.core.models import ClaimVerification, EvidenceItem

_NS = uuid.UUID("6f1c2a5e-8f1d-4c43-9a55-0c7a3b2d1e90")
AUTHOR_NAME = "whentrue-cti"


def _det_id(sdo_type: str, key: str) -> str:
    return f"{sdo_type}--{uuid.uuid5(_NS, f'{sdo_type}:{key.lower()}')}"


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _ext_refs(items: Iterable[EvidenceItem]):
    import stix2

    refs, seen = [], set()
    for e in items:
        if e.evidence_id in seen:
            continue
        seen.add(e.evidence_id)
        refs.append(
            stix2.ExternalReference(
                source_name=(e.publisher or "unknown publisher")[:200],
                description=f"report {e.report_id or 'n/a'}; published {e.pub_date or 'undated'}; source_type {e.source_type}",
                external_id=e.evidence_id,
            )
        )
    return refs


def _sdo(kind: str, name: str, author_id: str):
    import stix2

    created = datetime(2024, 1, 1, tzinfo=timezone.utc)
    common = dict(id=_det_id(kind, name), created=created, modified=created, created_by_ref=author_id)
    if kind == "threat-actor":
        return stix2.ThreatActor(name=name, **common)
    if kind == "malware":
        return stix2.Malware(name=name, is_family=True, **common)
    if kind == "tool":
        return stix2.Tool(name=name, **common)
    if kind == "vulnerability":
        return stix2.Vulnerability(name=name, **common)
    if kind == "attack-pattern":
        return stix2.AttackPattern(name=name, **common)
    return stix2.Identity(name=name, identity_class="class", **common)


def _kinds(v: ClaimVerification, aliases: AliasResolver) -> tuple:
    subj, obj = aliases.resolve(v.claim.subject), aliases.resolve(v.claim.object)
    cat = {"ThreatActor": "threat-actor", "Malware": "malware", "Tool": "tool"}
    s_kind = cat.get(subj.category or "", "threat-actor")
    if v.claim.relation == "EXPLOITS":
        o_kind = "vulnerability"
    elif v.claim.relation == "USES_TECHNIQUE":
        o_kind = "attack-pattern"
    elif v.claim.relation == "TARGETS":
        o_kind = "identity"
    else:
        o_kind = cat.get(obj.category or "", "malware")
    s_name = subj.canonical_name if subj.in_registry else v.claim.subject
    o_name = obj.canonical_name if obj.in_registry else v.claim.object
    return (s_kind, s_name), (o_kind, o_name)


def build_bundle(verifications: List[ClaimVerification], aliases: AliasResolver) -> Dict[str, Any]:
    import json

    import stix2

    created = datetime(2024, 1, 1, tzinfo=timezone.utc)
    author = stix2.Identity(
        id=_det_id("identity", AUTHOR_NAME), name=AUTHOR_NAME, identity_class="system", created=created, modified=created
    )
    objects: Dict[str, Any] = {author.id: author}
    report_refs: List[str] = []
    latest: Optional[datetime] = None

    for v in verifications:
        (s_kind, s_name), (o_kind, o_name) = _kinds(v, aliases)
        s_obj, o_obj = _sdo(s_kind, s_name, author.id), _sdo(o_kind, o_name, author.id)
        objects[s_obj.id], objects[o_obj.id] = s_obj, o_obj
        evidence = list(v.supporting) + list(v.contradicting)
        for e in evidence:
            d = _parse_date(e.pub_date)
            if d and (latest is None or d > latest):
                latest = d
        refs_for_note = [s_obj.id, o_obj.id]
        if v.status == "SUPPORTED" and v.polarity == "positive":
            rel_type = "targets" if v.claim.relation in ("TARGETS", "EXPLOITS") else "uses"
            rel = stix2.Relationship(
                id=_det_id("relationship", f"{s_obj.id}|{rel_type}|{o_obj.id}"),
                relationship_type=rel_type, source_ref=s_obj.id, target_ref=o_obj.id,
                description=f"{v.claim.subject} {v.claim.relation} {v.claim.object} (support tier {v.support_tier})",
                external_references=_ext_refs(v.supporting) or None,
                created=created, modified=created, created_by_ref=author.id,
            )
            objects[rel.id] = rel
            refs_for_note.append(rel.id)
            report_refs.append(rel.id)
        note = stix2.Note(
            id=_det_id("note", v.verification_id),
            abstract=f"Verification of: {v.claim.subject} {v.claim.relation} {v.claim.object}",
            content=(
                f"status={v.status}; support_tier={v.support_tier}; polarity={v.polarity}; "
                f"supporting={len(v.supporting)}; contradicting={len(v.contradicting)}; "
                f"conflict={v.conflict.type}; verification_id={v.verification_id}"
            ),
            object_refs=refs_for_note,
            external_references=_ext_refs(evidence) or None,
            created=created, modified=created, created_by_ref=author.id,
        )
        objects[note.id] = note
        report_refs.append(note.id)

    published = latest or datetime.now(timezone.utc)
    report = stix2.Report(
        id=_det_id("report", ",".join(sorted(v.verification_id for v in verifications))),
        name="whentrue-cti verification export",
        report_types=["threat-report"], published=published,
        object_refs=sorted(set(report_refs)) or [author.id],
        created=created, modified=created, created_by_ref=author.id,
    )
    objects[report.id] = report
    bundle = stix2.Bundle(objects=list(objects.values()), allow_custom=False)
    data = json.loads(bundle.serialize())

    try:
        from stix2validator import validate_string

        result = validate_string(json.dumps(data), options=None)
        if not result.is_valid:
            msgs = "; ".join(str(e) for e in getattr(result, "errors", [])[:3]) if hasattr(result, "errors") else "invalid"
            raise InternalError(f"Generated STIX bundle failed validation: {msgs}")
    except ImportError:  # pragma: no cover
        pass
    return data
