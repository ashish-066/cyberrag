"""Cybersecurity Knowledge Gazetteer and Alias Mappings.

Contains canonical entities, aliases, MITRE ATT&CK codes, and classifications
for Threat Actors, Malware, Tools, CVEs, Techniques, Organizations, and Locations.
"""

from typing import Dict, List, Any
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Threat Actors: Canonical Name -> Details & Aliases
THREAT_ACTORS: Dict[str, Dict[str, Any]] = {
    "Volt Typhoon": {
        "canonical_id": "TA-CN-0001",
        "mitre_id": "G0100",
        "origin": "China",
        "aliases": ["Volt Typhoon", "Vanguard Panda", "Bronze Silhouette", "Insidious Taurus", "Dev-0391"],
    },
    "APT29": {
        "canonical_id": "TA-RU-0002",
        "mitre_id": "G0016",
        "origin": "Russia",
        "aliases": ["APT29", "Cozy Bear", "Nobelium", "Midnight Blizzard", "The Dukes", "YTTRIUM", "Iron Hemlock"],
    },
    "APT28": {
        "canonical_id": "TA-RU-0001",
        "mitre_id": "G0007",
        "origin": "Russia",
        "aliases": ["APT28", "Fancy Bear", "Strontium", "Forest Blizzard", "Sednit", "Pawn Storm", "Sofacy", "Tsar Team"],
    },
    "Lazarus Group": {
        "canonical_id": "TA-KP-0001",
        "mitre_id": "G0032",
        "origin": "North Korea",
        "aliases": ["Lazarus Group", "Hidden Cobra", "Zinc", "Guardians of Peace", "Labyrinth Chollima", "Diamond Sleet"],
    },
    "Sandworm": {
        "canonical_id": "TA-RU-0003",
        "mitre_id": "G0034",
        "origin": "Russia",
        "aliases": ["Sandworm", "Voodoo Bear", "APT44", "FROZENBARENTS", "TeleBots", "BlackEnergy Group", "Iron Viking", "Seashell Blizzard"],
    },
    "LockBit": {
        "canonical_id": "TA-CRIME-0001",
        "mitre_id": "G0140",
        "origin": "Financially Motivated",
        "aliases": ["LockBit", "LockBit 3.0", "LockBit Black", "LockBit Red", "LockBit affiliates", "LockBit Gang"],
    },
    "FIN7": {
        "canonical_id": "TA-CRIME-0002",
        "mitre_id": "G0046",
        "origin": "Financially Motivated",
        "aliases": ["FIN7", "Carbanak", "Elbrus", "Sangria Tempest", "Gold Niagara"],
    },
    "Scattered Spider": {
        "canonical_id": "TA-CRIME-0003",
        "mitre_id": "G1015",
        "origin": "Financially Motivated",
        "aliases": ["Scattered Spider", "Octo Tempest", "UNC3944", "Scatter Swine", "Starfraud"],
    },
    "Turla": {
        "canonical_id": "TA-RU-0004",
        "mitre_id": "G0010",
        "origin": "Russia",
        "aliases": ["Turla", "Waterbug", "Venomous Bear", "KRYPTON", "Secret Blizzard"],
    },
    "Charming Kitten": {
        "canonical_id": "TA-IR-0001",
        "mitre_id": "G0058",
        "origin": "Iran",
        "aliases": ["Charming Kitten", "APT35", "Phosphorus", "Mint Sandstorm", "Newscaster"],
    },
}

# Malware Families: Canonical Name -> Details & Aliases
MALWARE: Dict[str, Dict[str, Any]] = {
    "AppleJeus": {
        "canonical_id": "MAL-0001",
        "mitre_id": "S0484",
        "type": "Trojan",
        "aliases": ["AppleJeus"],
    },
    "MagicRAT": {
        "canonical_id": "MAL-0002",
        "mitre_id": "S1070",
        "type": "Remote Access Trojan",
        "aliases": ["MagicRAT"],
    },
    "HermeticWiper": {
        "canonical_id": "MAL-0003",
        "mitre_id": "S0697",
        "type": "Wiper",
        "aliases": ["HermeticWiper", "KillDisk.NCV"],
    },
    "CaddyWiper": {
        "canonical_id": "MAL-0004",
        "mitre_id": "S1014",
        "type": "Wiper",
        "aliases": ["CaddyWiper"],
    },
    "Industroyer2": {
        "canonical_id": "MAL-0005",
        "mitre_id": "S1018",
        "type": "ICS Malware",
        "aliases": ["Industroyer2", "CrashOverride2"],
    },
    "BlackCat": {
        "canonical_id": "MAL-0006",
        "mitre_id": "S1068",
        "type": "Ransomware",
        "aliases": ["BlackCat", "ALPHV", "BlackCat ransomware"],
    },
    "Carbanak": {
        "canonical_id": "MAL-0007",
        "mitre_id": "S0030",
        "type": "Banking Trojan",
        "aliases": ["Carbanak malware", "Anunak"],
    },
    "POWERPLANT": {
        "canonical_id": "MAL-0008",
        "mitre_id": "S0680",
        "type": "Backdoor",
        "aliases": ["POWERPLANT", "Powerplant backdoor"],
    },
    "TrickBot": {
        "canonical_id": "MAL-0009",
        "mitre_id": "S0266",
        "type": "Modular Trojan",
        "aliases": ["TrickBot", "TheTrick"],
    },
    "Emotet": {
        "canonical_id": "MAL-0010",
        "mitre_id": "S0367",
        "type": "Botnet/Loader",
        "aliases": ["Emotet", "Geodo"],
    },
    "Cobalt Strike BEACON": {
        "canonical_id": "MAL-0011",
        "mitre_id": "S0154",
        "type": "Payload/Beacon",
        "aliases": ["Cobalt Strike BEACON", "BEACON", "Cobalt Strike payload"],
    },
    "KV-Botnet": {
        "canonical_id": "MAL-0012",
        "mitre_id": "S1120",
        "type": "Botnet",
        "aliases": ["KV-Botnet", "KV botnet"],
    },
}

# Tools: Canonical Name -> Details & Aliases
TOOLS: Dict[str, Dict[str, Any]] = {
    "Cobalt Strike": {
        "canonical_id": "TOOL-0001",
        "mitre_id": "S0154",
        "category": "C2 Framework",
        "aliases": ["Cobalt Strike", "Cobalt Strike framework"],
    },
    "Mimikatz": {
        "canonical_id": "TOOL-0002",
        "mitre_id": "S0002",
        "category": "Credential Access",
        "aliases": ["Mimikatz", "mimidrv"],
    },
    "PsExec": {
        "canonical_id": "TOOL-0003",
        "mitre_id": "S0029",
        "category": "Execution / Remote Administration",
        "aliases": ["PsExec", "Sysinternals PsExec"],
    },
    "Chisel": {
        "canonical_id": "TOOL-0004",
        "mitre_id": "S0553",
        "category": "Tunneling / Proxy",
        "aliases": ["Chisel", "Chisel proxy", "chisel.exe"],
    },
    "Fast Reverse Proxy": {
        "canonical_id": "TOOL-0005",
        "mitre_id": "S0647",
        "category": "Tunneling / Proxy",
        "aliases": ["Fast Reverse Proxy", "FRP", "frpc", "frps"],
    },
    "PowerShell": {
        "canonical_id": "TOOL-0006",
        "mitre_id": "T1059.001",
        "category": "Living-off-the-Land / Scripting",
        "aliases": ["PowerShell", "powershell.exe", "PowerShell scripts"],
    },
    "WMI": {
        "canonical_id": "TOOL-0007",
        "mitre_id": "T1047",
        "category": "Living-off-the-Land / Management",
        "aliases": ["WMI", "Windows Management Instrumentation", "wmic.exe"],
    },
    "ntdsutil": {
        "canonical_id": "TOOL-0008",
        "mitre_id": "T1003.003",
        "category": "Living-off-the-Land / Credential Harvesting",
        "aliases": ["ntdsutil", "ntdsutil.exe"],
    },
    "PuTTY": {
        "canonical_id": "TOOL-0009",
        "mitre_id": "S0692",
        "category": "Remote Access / SSH",
        "aliases": ["PuTTY", "Plink", "putty.exe"],
    },
    "BloodHound": {
        "canonical_id": "TOOL-0010",
        "mitre_id": "S0521",
        "category": "Reconnaissance / Discovery",
        "aliases": ["BloodHound", "SharpHound"],
    },
    "Impacket": {
        "canonical_id": "TOOL-0011",
        "mitre_id": "S0357",
        "category": "Network / Lateral Movement",
        "aliases": ["Impacket", "Impacket toolsets", "wmiexec", "psexec.py"],
    },
    "AnyDesk": {
        "canonical_id": "TOOL-0012",
        "mitre_id": "S0642",
        "category": "Remote Desktop Software",
        "aliases": ["AnyDesk", "AnyDesk remote access"],
    },
    "MegaSync": {
        "canonical_id": "TOOL-0013",
        "mitre_id": "S0625",
        "category": "Data Exfiltration / Cloud Storage",
        "aliases": ["MegaSync", "MEGA client"],
    },
    "rclone": {
        "canonical_id": "TOOL-0014",
        "mitre_id": "S0544",
        "category": "Data Exfiltration / Cloud Sync",
        "aliases": ["rclone", "rclone.exe"],
    },
    "Nmap": {
        "canonical_id": "TOOL-0015",
        "mitre_id": "S0039",
        "category": "Port Scanning / Discovery",
        "aliases": ["Nmap", "Network Mapper"],
    },
}

# Known Named Vulnerabilities & Famous CVEs
KNOWN_CVES: Dict[str, Dict[str, Any]] = {
    "CVE-2024-21887": {
        "canonical_id": "CVE-2024-21887",
        "name": "Ivanti Connect Secure Command Injection",
        "affected_product": "Ivanti Connect Secure / Policy Secure",
        "aliases": ["CVE-2024-21887", "Ivanti command injection"],
    },
    "CVE-2023-46805": {
        "canonical_id": "CVE-2023-46805",
        "name": "Ivanti Connect Secure Authentication Bypass",
        "affected_product": "Ivanti Connect Secure",
        "aliases": ["CVE-2023-46805", "Ivanti auth bypass"],
    },
    "CVE-2023-4966": {
        "canonical_id": "CVE-2023-4966",
        "name": "Citrix Bleed Sensitive Information Disclosure",
        "affected_product": "Citrix NetScaler ADC / Gateway",
        "aliases": ["CVE-2023-4966", "Citrix Bleed"],
    },
    "CVE-2023-38831": {
        "canonical_id": "CVE-2023-38831",
        "name": "WinRAR Code Execution Vulnerability",
        "affected_product": "WinRAR",
        "aliases": ["CVE-2023-38831", "WinRAR flaw"],
    },
    "CVE-2021-44228": {
        "canonical_id": "CVE-2021-44228",
        "name": "Apache Log4j Remote Code Execution (Log4Shell)",
        "affected_product": "Apache Log4j",
        "aliases": ["CVE-2021-44228", "Log4Shell", "Log4j vulnerability"],
    },
    "CVE-2022-47966": {
        "canonical_id": "CVE-2022-47966",
        "name": "Zoho ManageEngine RCE Vulnerability",
        "affected_product": "Zoho ManageEngine ServiceDesk Plus",
        "aliases": ["CVE-2022-47966", "ManageEngine RCE"],
    },
    "CVE-2022-30190": {
        "canonical_id": "CVE-2022-30190",
        "name": "Microsoft MSDT Remote Code Execution (Follina)",
        "affected_product": "Microsoft Windows MSDT",
        "aliases": ["CVE-2022-30190", "Follina", "MSDT vulnerability"],
    },
    "CVE-2021-34527": {
        "canonical_id": "CVE-2021-34527",
        "name": "Windows Print Spooler RCE (PrintNightmare)",
        "affected_product": "Windows Print Spooler",
        "aliases": ["CVE-2021-34527", "PrintNightmare"],
    },
    "CVE-2024-3400": {
        "canonical_id": "CVE-2024-3400",
        "name": "Palo Alto Networks PAN-OS Command Injection",
        "affected_product": "Palo Alto PAN-OS",
        "aliases": ["CVE-2024-3400", "PAN-OS vulnerability"],
    },
    "CVE-2023-34362": {
        "canonical_id": "CVE-2023-34362",
        "name": "MOVEit Transfer SQL Injection",
        "affected_product": "Progress MOVEit Transfer",
        "aliases": ["CVE-2023-34362", "MOVEit flaw", "MOVEit vulnerability"],
    },
    "CVE-2020-1472": {
        "canonical_id": "CVE-2020-1472",
        "name": "Netlogon Elevation of Privilege (Zerologon)",
        "affected_product": "Microsoft Windows Netlogon",
        "aliases": ["CVE-2020-1472", "Zerologon"],
    },
    "CVE-2017-0144": {
        "canonical_id": "CVE-2017-0144",
        "name": "Windows SMB Remote Code Execution (EternalBlue)",
        "affected_product": "Microsoft Windows SMBv1",
        "aliases": ["CVE-2017-0144", "EternalBlue"],
    },
}

# MITRE ATT&CK Techniques: ID / Name -> Canonical Entry
TECHNIQUES: Dict[str, Dict[str, Any]] = {
    "T1566": {
        "canonical_id": "T1566",
        "name": "Phishing",
        "tactic": "Initial Access",
        "aliases": ["T1566", "Phishing", "spearphishing", "spearphishing emails", "malicious attachments", "lure delivery"],
    },
    "T1190": {
        "canonical_id": "T1190",
        "name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
        "aliases": ["T1190", "Exploit Public-Facing Application", "exploiting public-facing applications", "edge device exploitation"],
    },
    "T1059": {
        "canonical_id": "T1059",
        "name": "Command and Scripting Interpreter",
        "tactic": "Execution",
        "aliases": ["T1059", "Command and Scripting Interpreter", "scripting interpreter", "living-off-the-land techniques", "living-off-the-land"],
    },
    "T1003": {
        "canonical_id": "T1003",
        "name": "OS Credential Dumping",
        "tactic": "Credential Access",
        "aliases": ["T1003", "OS Credential Dumping", "credential dumping", "harvest credentials", "extract administrator credentials", "dump plaintext credentials"],
    },
    "T1078": {
        "canonical_id": "T1078",
        "name": "Valid Accounts",
        "tactic": "Defense Evasion / Initial Access",
        "aliases": ["T1078", "Valid Accounts", "compromised valid accounts", "hijacked credentials"],
    },
    "T1485": {
        "canonical_id": "T1485",
        "name": "Data Destruction",
        "tactic": "Impact",
        "aliases": ["T1485", "Data Destruction", "destructive malware", "overwrite master boot records"],
    },
    "T1486": {
        "canonical_id": "T1486",
        "name": "Data Encrypted for Impact",
        "tactic": "Impact",
        "aliases": ["T1486", "Data Encrypted for Impact", "ransomware encryption", "extortion phases"],
    },
    "T1021": {
        "canonical_id": "T1021",
        "name": "Remote Services",
        "tactic": "Lateral Movement",
        "aliases": ["T1021", "Remote Services", "lateral movement", "remote endpoints execution"],
    },
    "T1574": {
        "canonical_id": "T1574",
        "name": "Hijack Execution Flow",
        "tactic": "Persistence / Privilege Escalation",
        "aliases": ["T1574", "Hijack Execution Flow", "DLL sideloading", "DLL search order hijacking"],
    },
}

# Organizations & Industry Sectors: Canonical Name -> Aliases
ORGANIZATIONS: Dict[str, Dict[str, Any]] = {
    "Microsoft": {
        "canonical_id": "ORG-0001",
        "type": "Technology Vendor",
        "aliases": ["Microsoft", "Microsoft enterprise accounts"],
    },
    "NATO": {
        "canonical_id": "ORG-0002",
        "type": "International Defense Organization",
        "aliases": ["NATO", "NATO member states"],
    },
    "Ukrainian Ministry of Defense": {
        "canonical_id": "ORG-0003",
        "type": "Government / Defense",
        "aliases": ["Ukrainian Ministry of Defense", "Ministry of Defense of Ukraine", "Ukrainian military"],
    },
    "Defense Industrial Base": {
        "canonical_id": "ORG-0004",
        "type": "Critical Infrastructure / Defense",
        "aliases": ["Defense Industrial Base", "defense contractors", "aerospace engineering firms"],
    },
    "Energy Sector": {
        "canonical_id": "ORG-0005",
        "type": "Critical Infrastructure",
        "aliases": ["Energy Sector", "Energy", "energy grid", "national electric substations", "regional energy distribution networks"],
    },
    "Communications Sector": {
        "canonical_id": "ORG-0006",
        "type": "Critical Infrastructure",
        "aliases": ["Communications Sector", "Communications", "telecommunications providers", "telecommunication providers"],
    },
    "Transportation Systems Sector": {
        "canonical_id": "ORG-0007",
        "type": "Critical Infrastructure",
        "aliases": ["Transportation Systems", "railway systems", "port authorities"],
    },
    "Healthcare Sector": {
        "canonical_id": "ORG-0008",
        "type": "Healthcare",
        "aliases": ["healthcare systems", "hospital networks", "Healthcare"],
    },
    "Financial Sector": {
        "canonical_id": "ORG-0009",
        "type": "Financial",
        "aliases": ["financial institutions", "cryptocurrency exchanges", "financial services firms", "cryptocurrency corporations"],
    },
    "Retail & Hospitality Sector": {
        "canonical_id": "ORG-0010",
        "type": "Commercial",
        "aliases": ["Retail & Hospitality Sector", "retail enterprises", "retail corporations", "restaurant franchises", "hospitality chains", "major hotel chains"],
    },
    "Municipal Governments": {
        "canonical_id": "ORG-0011",
        "type": "Government",
        "aliases": ["Municipal Governments", "municipal governments", "government agencies", "European diplomatic entities", "diplomatic entities"],
    },
    "Ukrainian Government Organizations": {
        "canonical_id": "ORG-0012",
        "type": "Government",
        "aliases": ["Ukrainian government organizations", "Ukrainian government agencies"],
    },
}

# Locations: Geopolitical Entities
LOCATIONS: Dict[str, Dict[str, Any]] = {
    "United States": {
        "canonical_id": "LOC-US",
        "iso": "USA",
        "aliases": ["United States", "U.S.", "US", "America"],
    },
    "Ukraine": {
        "canonical_id": "LOC-UA",
        "iso": "UKR",
        "aliases": ["Ukraine", "Ukrainian"],
    },
    "Guam": {
        "canonical_id": "LOC-GU",
        "iso": "GUM",
        "aliases": ["Guam"],
    },
    "Hawaii": {
        "canonical_id": "LOC-US-HI",
        "iso": "USA-HI",
        "aliases": ["Hawaii"],
    },
    "Poland": {
        "canonical_id": "LOC-PL",
        "iso": "POL",
        "aliases": ["Poland"],
    },
    "South Korea": {
        "canonical_id": "LOC-KR",
        "iso": "KOR",
        "aliases": ["South Korea", "Republic of Korea"],
    },
    "Japan": {
        "canonical_id": "LOC-JP",
        "iso": "JPN",
        "aliases": ["Japan"],
    },
    "United Kingdom": {
        "canonical_id": "LOC-GB",
        "iso": "GBR",
        "aliases": ["United Kingdom", "UK", "Britain"],
    },
    "Germany": {
        "canonical_id": "LOC-DE",
        "iso": "DEU",
        "aliases": ["Germany"],
    },
    "Canada": {
        "canonical_id": "LOC-CA",
        "iso": "CAN",
        "aliases": ["Canada"],
    },
    "Europe": {
        "canonical_id": "LOC-EU",
        "iso": "EUR",
        "aliases": ["Europe", "European"],
    },
    "Eastern Europe": {
        "canonical_id": "LOC-EEU",
        "iso": "EEU",
        "aliases": ["Eastern Europe"],
    },
    "China": {
        "canonical_id": "LOC-CN",
        "iso": "CHN",
        "aliases": ["China", "Chinese", "People's Republic of China"],
    },
    "Russia": {
        "canonical_id": "LOC-RU",
        "iso": "RUS",
        "aliases": ["Russia", "Russian"],
    },
    "North Korea": {
        "canonical_id": "LOC-KP",
        "iso": "PRK",
        "aliases": ["North Korea", "DPRK"],
    },
    "Taiwan": {
        "canonical_id": "LOC-TW",
        "iso": "TWN",
        "aliases": ["Taiwan", "Republic of China"],
    },
    "Iran": {
        "canonical_id": "LOC-IR",
        "iso": "IRN",
        "aliases": ["Iran", "Islamic Republic of Iran", "Iranian"],
    },
    "France": {
        "canonical_id": "LOC-FR",
        "iso": "FRA",
        "aliases": ["France", "French"],
    },
    "Netherlands": {
        "canonical_id": "LOC-NL",
        "iso": "NLD",
        "aliases": ["Netherlands", "Holland", "Dutch"],
    },
    "Australia": {
        "canonical_id": "LOC-AU",
        "iso": "AUS",
        "aliases": ["Australia", "Australian"],
    },
    "Israel": {
        "canonical_id": "LOC-IL",
        "iso": "ISR",
        "aliases": ["Israel", "Israeli"],
    },
    "India": {
        "canonical_id": "LOC-IN",
        "iso": "IND",
        "aliases": ["India", "Indian"],
    },
    "Singapore": {
        "canonical_id": "LOC-SG",
        "iso": "SGP",
        "aliases": ["Singapore"],
    },
    "Middle East": {
        "canonical_id": "LOC-ME",
        "iso": "MEA",
        "aliases": ["Middle East"],
    },
    "Asia-Pacific": {
        "canonical_id": "LOC-APAC",
        "iso": "APAC",
        "aliases": ["Asia-Pacific", "APAC"],
    },
    "North America": {
        "canonical_id": "LOC-NA",
        "iso": "NAM",
        "aliases": ["North America"],
    },
}


def _load_mitre_registry():
    """Dynamically load and merge official MITRE Enterprise ATT&CK groups and software."""
    registry_path = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "mitre_attack_registry.json"
    if not registry_path.exists():
        registry_path = Path("data/knowledge/mitre_attack_registry.json")
    if not registry_path.exists():
        return

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        # Merge threat actors
        for name, data in registry.get("threat_actors", {}).items():
            if name in THREAT_ACTORS:
                existing_aliases = set(THREAT_ACTORS[name].get("aliases", []))
                for a in data.get("aliases", []):
                    existing_aliases.add(a)
                THREAT_ACTORS[name]["aliases"] = sorted(list(existing_aliases), key=len, reverse=True)
            else:
                THREAT_ACTORS[name] = data

        # Merge malware
        for name, data in registry.get("malware", {}).items():
            if name in TOOLS:
                continue
            if name in MALWARE:
                existing_aliases = set(MALWARE[name].get("aliases", []))
                for a in data.get("aliases", []):
                    existing_aliases.add(a)
                MALWARE[name]["aliases"] = sorted(list(existing_aliases), key=len, reverse=True)
            else:
                MALWARE[name] = data

        # Merge tools
        for name, data in registry.get("tools", {}).items():
            if name in TOOLS:
                existing_aliases = set(TOOLS[name].get("aliases", []))
                for a in data.get("aliases", []):
                    existing_aliases.add(a)
                TOOLS[name]["aliases"] = sorted(list(existing_aliases), key=len, reverse=True)
            else:
                TOOLS[name] = data

        # Merge techniques
        for tech_id, data in registry.get("techniques", {}).items():
            if tech_id in TECHNIQUES:
                existing_aliases = set(TECHNIQUES[tech_id].get("aliases", []))
                for a in data.get("aliases", []):
                    existing_aliases.add(a)
                TECHNIQUES[tech_id]["aliases"] = sorted(list(existing_aliases), key=len, reverse=True)
            else:
                TECHNIQUES[tech_id] = data

        logger.debug(f"Loaded MITRE registry: {len(THREAT_ACTORS)} actors, {len(MALWARE)} malware, {len(TOOLS)} tools.")
    except Exception as exc:
        logger.warning(f"Failed to load MITRE ATT&CK registry: {exc}")


_load_mitre_registry()

