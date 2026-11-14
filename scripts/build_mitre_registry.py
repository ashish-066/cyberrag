"""MITRE ATT&CK Enterprise Knowledge Base Builder.

Fetches the official STIX 2.1 Enterprise ATT&CK bundle from MITRE CTI repository,
extracts all Intrusion Sets (Threat Actors), Malware, Tools, and Techniques,
normalizes aliases, filters stopwords to avoid false-positive NER collisions,
and compiles a production knowledge registry for the CyberRAG knowledge graph.
"""

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Dict, Any, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("MITREBuilder")

MITRE_ENTERPRISE_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
OUTPUT_PATH = Path("data/knowledge/mitre_attack_registry.json")

# Common English words or noisy abbreviations that appear as aliases in MITRE but cause massive FP in NLP
STOPWORDS_BLACKLIST: Set[str] = {
    "at", "all", "and", "by", "for", "in", "is", "it", "of", "on", "or", "the", "to", "with",
    "as", "be", "do", "go", "if", "me", "my", "no", "so", "up", "we", "he", "an", "am", "us",
    "him", "her", "his", "how", "why", "who", "what", "when", "down", "out", "left", "right",
    "back", "drop", "pass", "patch", "start", "stop", "step", "test", "type", "user", "file",
    "host", "port", "read", "send", "time", "date", "task", "unit", "system", "group", "actor",
    "threat", "cyber", "malware", "tool", "target", "attack", "action", "service", "client",
    "server", "proxy", "agent", "data", "base", "net", "web", "ping", "find", "whoami", "ipconfig",
    "certutil", "curl", "cmd", "powershell", "reg", "sc", "tasklist", "netstat", "route", "ftp",
    "arp", "ps1", "bbk", "abk", "tvt", "orz", "yty", "dok", "zox", "tor", "wce", "frp"
}


def clean_alias(alias: str) -> str:
    """Normalize alias surface text and strip unwanted formatting."""
    alias = alias.strip().strip("'\"()[]{}")
    return alias


def is_valid_alias(alias: str) -> bool:
    """Check if alias is sufficiently specific for NER gazetteer matching."""
    if not alias or len(alias) < 3:
        return False
    if alias.lower() in STOPWORDS_BLACKLIST:
        return False
    # Pure numbers are not valid names (e.g. '01', '360')
    if alias.isdigit():
        return False
    # If 3 characters, require it to be uppercase acronym (e.g. RTM, UPX)
    if len(alias) <= 3 and not alias.isupper():
        return False
    return True


def build_registry():
    logger.info(f"Fetching MITRE Enterprise ATT&CK bundle from {MITRE_ENTERPRISE_URL}...")
    req = urllib.request.Request(MITRE_ENTERPRISE_URL, headers={"User-Agent": "CyberRAG-CTI-Ingestion/2.0"})
    
    with urllib.request.urlopen(req, timeout=45) as resp:
        stix_bundle = json.loads(resp.read().decode("utf-8"))

    objects = stix_bundle.get("objects", [])
    logger.info(f"Loaded {len(objects)} STIX objects. Parsing categories...")

    threat_actors: Dict[str, Dict[str, Any]] = {}
    malware: Dict[str, Dict[str, Any]] = {}
    tools: Dict[str, Dict[str, Any]] = {}
    techniques: Dict[str, Dict[str, Any]] = {}

    for obj in objects:
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        obj_type = obj.get("type")
        name = obj.get("name", "").strip()
        if not name:
            continue

        # Extract MITRE ID (e.g. G0016, S0154, T1059)
        mitre_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") in ("mitre-attack", "mitre-enterprise-attack"):
                mitre_id = ref.get("external_id", "")
                break

        if not mitre_id:
            continue

        description = obj.get("description", "")
        # First paragraph of description
        desc_summary = description.split("\n\n")[0] if description else ""

        # 1. Intrusion Sets -> Threat Actors
        if obj_type == "intrusion-set":
            aliases = set()
            aliases.add(name)
            for a in obj.get("aliases", []):
                cleaned = clean_alias(a)
                if is_valid_alias(cleaned):
                    aliases.add(cleaned)

            threat_actors[name] = {
                "canonical_id": f"TA-MITRE-{mitre_id}",
                "mitre_id": mitre_id,
                "name": name,
                "description": desc_summary[:300],
                "aliases": sorted(list(aliases), key=len, reverse=True),
            }

        # 2. Malware
        elif obj_type == "malware":
            aliases = set()
            aliases.add(name)
            raw_aliases = obj.get("aliases", []) + obj.get("x_mitre_aliases", [])
            for a in raw_aliases:
                cleaned = clean_alias(a)
                if is_valid_alias(cleaned):
                    aliases.add(cleaned)

            malware[name] = {
                "canonical_id": f"MAL-MITRE-{mitre_id}",
                "mitre_id": mitre_id,
                "name": name,
                "description": desc_summary[:300],
                "aliases": sorted(list(aliases), key=len, reverse=True),
            }

        # 3. Tool
        elif obj_type == "tool":
            aliases = set()
            aliases.add(name)
            raw_aliases = obj.get("aliases", []) + obj.get("x_mitre_aliases", [])
            for a in raw_aliases:
                cleaned = clean_alias(a)
                if is_valid_alias(cleaned):
                    aliases.add(cleaned)

            tools[name] = {
                "canonical_id": f"TOOL-MITRE-{mitre_id}",
                "mitre_id": mitre_id,
                "name": name,
                "description": desc_summary[:300],
                "aliases": sorted(list(aliases), key=len, reverse=True),
            }

        # 4. Technique (attack-pattern)
        elif obj_type == "attack-pattern":
            aliases = set([name, mitre_id])
            for a in obj.get("x_mitre_aliases", []):
                cleaned = clean_alias(a)
                if is_valid_alias(cleaned):
                    aliases.add(cleaned)

            techniques[mitre_id] = {
                "canonical_id": f"TECH-{mitre_id}",
                "mitre_id": mitre_id,
                "name": name,
                "tactic": [phase.get("phase_name") for phase in obj.get("kill_chain_phases", [])],
                "aliases": sorted(list(aliases), key=len, reverse=True),
            }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        "metadata": {
            "source": "MITRE Enterprise ATT&CK STIX 2.1",
            "url": MITRE_ENTERPRISE_URL,
            "threat_actors_count": len(threat_actors),
            "malware_count": len(malware),
            "tools_count": len(tools),
            "techniques_count": len(techniques),
        },
        "threat_actors": threat_actors,
        "malware": malware,
        "tools": tools,
        "techniques": techniques,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    logger.info("=" * 60)
    logger.info("MITRE ATT&CK REGISTRY GENERATED SUCCESSFULLY:")
    logger.info(f"  - Threat Actors (Intrusion Sets): {len(threat_actors)}")
    logger.info(f"  - Malware Families:               {len(malware)}")
    logger.info(f"  - Tools:                          {len(tools)}")
    logger.info(f"  - Techniques & Subtechniques:     {len(techniques)}")
    logger.info(f"Saved registry to: {OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    build_registry()
