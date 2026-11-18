"""
Query and Alias Expansion for CyberRAG.
Leverages curated MITRE ATT&CK and CTI threat actor / malware alias registries.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

# Curated CTI Threat Actor & Malware Aliases (MITRE ATT&CK + AnnoCTR corpus)
KNOWN_ALIASES: Dict[str, List[str]] = {
    "sandworm": ["telebots", "voodoo bear", "iron viking", "unit 74455", "blackenergy"],
    "emotet": ["geodo", "heodo"],
    "trickbot": ["theb trickbot", "wizard spider", "anchor"],
    "apt28": ["fancy bear", "strontium", "sednit", "pawn storm", "sofacy"],
    "apt29": ["cozy bear", "the dukes", "nobelium"],
    "lazarus": ["lazarus group", "hidden cobra", "zinc", "guardians of peace"],
    "lazarus group": ["lazarus", "hidden cobra", "zinc", "guardians of peace"],
    "fin7": ["carbanak", "navigator group"],
    "turla": ["waterbug", "venomous bear", "snake", "krypton"],
    "dragonfly": ["energetic bear", "crouching yeti"],
    "oilrig": ["helix kitten", "apt34", "chrysene"],
    "taurus": ["taurus stealer"],
    "blackcat": ["alphv"],
    "cobalt strike": ["cobaltstrike", "beacon"],
    "qakbot": ["qbot", "quakbot"],
    "conti": ["wizard spider"],
    "banking": ["banks", "banking organizations", "financial sector", "banking fraud", "financial institutions"],
    "banking organizations": ["banking", "banks", "financial sector", "banking fraud", "financial institutions"],
    "banks": ["banking", "banking organizations", "financial sector"],
    "hospitality": ["hospitality sector", "hotel chains", "restaurants"],
    "government": ["government agencies", "public sector", "ministries"],
}

# Generic words that must NEVER independently satisfy entity presence
GENERIC_ENTITY_STOPS = {
    "group", "groups", "team", "teams", "unit", "units", "threat", "threats",
    "actor", "actors", "campaign", "campaigns", "operation", "operations",
    "malware", "tool", "tools", "trojan", "loader", "stealer", "backdoor",
    "ransomware", "virus", "worm", "apt", "corp", "corporation", "inc", "ltd",
    "organization", "organizations", "org", "orgs", "agency", "agencies",
    "company", "companies", "sector", "sectors", "industry", "industries",
    "government", "governments", "target", "targets", "targeted", "targeting",
    "associated", "related", "the", "and", "for", "with", "from"
}


class QueryExpander:
    """Expands queries using domain aliases and multi-hop search variations."""

    def __init__(self, alias_dict: Optional[Dict[str, List[str]]] = None):
        self.alias_dict = alias_dict or KNOWN_ALIASES

    def get_aliases(self, entity: str) -> List[str]:
        return self.alias_dict.get(entity.lower(), [])

    def get_search_terms(self, entity: str) -> Set[str]:
        e_lower = entity.lower().strip()
        terms: Set[str] = {e_lower}
        terms.add(e_lower.replace(" ", ""))
        terms.add(e_lower.replace("-", ""))

        # Extract distinctive root by stripping generic entity affixes
        root = re.sub(
            r"\b(group|groups|team|teams|unit|units|threat\s+actor|actor|actors|apt|campaign|operation|organizations|organization|orgs|org|sectors|sector|industries|industry|enterprises|enterprise|corporations|corporation|institutions|institution)\b",
            "",
            e_lower,
        ).strip()
        if len(root) >= 3 and root not in GENERIC_ENTITY_STOPS:
            terms.add(root)
            terms.add(root.replace(" ", ""))
            terms.add(root.replace("-", ""))

        for al in self.get_aliases(entity):
            al_lower = al.lower().strip()
            terms.add(al_lower)
            terms.add(al_lower.replace(" ", ""))
            terms.add(al_lower.replace("-", ""))
            al_root = re.sub(
                r"\b(group|groups|team|teams|unit|units|threat\s+actor|actor|actors|apt|campaign|operation|organizations|organization|orgs|org|sectors|sector|industries|industry|enterprises|enterprise|corporations|corporation|institutions|institution)\b",
                "",
                al_lower,
            ).strip()
            if len(al_root) >= 3 and al_root not in GENERIC_ENTITY_STOPS:
                terms.add(al_root)

        # Strictly exclude generic stop words to eliminate false-positive entity gate passes
        return {t for t in terms if t not in GENERIC_ENTITY_STOPS and len(t) >= 3}

    def expand_query(self, subject: str, obj: str, iteration: int) -> str:
        s_aliases = self.get_aliases(subject)
        o_aliases = self.get_aliases(obj)

        if iteration == 1 and s_aliases:
            return f"{s_aliases[0]} {obj}"
        elif iteration == 2 and o_aliases:
            return f"{subject} {o_aliases[0]}"
        elif s_aliases and o_aliases:
            return f"{s_aliases[0]} {o_aliases[0]}"
        return f"{subject} cyber threat actor malware {obj}"
