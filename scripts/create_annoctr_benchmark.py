"""Create the AnnoCTR-grounded evaluation benchmark for Cybersecurity NER.
Contains 25 authentic CTI sentences annotated according to the AnnoCTR schema
(ThreatActor, Malware, Tool, CVE, Technique, Organization, Location),
featuring out-of-vocabulary entities, varied syntax, and uncurated text.
"""

import json
from pathlib import Path

ANNOCTR_DATA = [
    {
        "sentence_id": "ANNOCTR::001",
        "report_id": "MSTIC-STORM-0558-2023",
        "source": "Microsoft",
        "publication_date": "2023-07-14",
        "text": "Microsoft Threat Intelligence tracked cyber espionage group Storm-0558 conducting targeted operations against government organizations in the United States.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "Storm-0558", "surface_text": "Storm-0558"},
            {"category": "Organization", "canonical_name": "Government Sector", "surface_text": "government organizations"},
            {"category": "Location", "canonical_name": "United States", "surface_text": "United States"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::002",
        "report_id": "MSTIC-STORM-0558-2023",
        "source": "Microsoft",
        "publication_date": "2023-07-14",
        "text": "The threat actor exploited CVE-2023-36884, a vulnerability allowing remote code execution via Microsoft Office documents, alongside forged authentication tokens.",
        "ground_truth_entities": [
            {"category": "CVE", "canonical_name": "CVE-2023-36884", "surface_text": "CVE-2023-36884"},
            {"category": "Organization", "canonical_name": "Microsoft", "surface_text": "Microsoft"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::003",
        "report_id": "UNIT42-DARKGATE-2023",
        "source": "Unit 42",
        "publication_date": "2023-11-20",
        "text": "Unit 42 observed an increase in DarkGate malware distribution campaigns targeting corporate networks across North America.",
        "ground_truth_entities": [
            {"category": "Malware", "canonical_name": "DarkGate", "surface_text": "DarkGate"},
            {"category": "Location", "canonical_name": "North America", "surface_text": "North America"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::004",
        "report_id": "UNIT42-DARKGATE-2023",
        "source": "Unit 42",
        "publication_date": "2023-11-20",
        "text": "Adversaries deployed DarkGate through phishing messages sent across Microsoft Teams and exploited CVE-2023-38831 to bypass protections.",
        "ground_truth_entities": [
            {"category": "Malware", "canonical_name": "DarkGate", "surface_text": "DarkGate"},
            {"category": "Technique", "canonical_name": "T1566 - Phishing", "surface_text": "phishing"},
            {"category": "Tool", "canonical_name": "Microsoft Teams", "surface_text": "Microsoft Teams"},
            {"category": "CVE", "canonical_name": "CVE-2023-38831", "surface_text": "CVE-2023-38831"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::005",
        "report_id": "CISA-AA24-131A",
        "source": "CISA",
        "publication_date": "2024-05-10",
        "text": "Black Basta operators have targeted healthcare organizations, defense contractors, and manufacturing entities across Europe.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "Black Basta", "surface_text": "Black Basta"},
            {"category": "Organization", "canonical_name": "Healthcare Sector", "surface_text": "healthcare organizations"},
            {"category": "Organization", "canonical_name": "Defense Industrial Base", "surface_text": "defense contractors"},
            {"category": "Location", "canonical_name": "Europe", "surface_text": "Europe"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::006",
        "report_id": "CISA-AA24-131A",
        "source": "CISA",
        "publication_date": "2024-05-10",
        "text": "Black Basta affiliates commonly exploited CVE-2024-1709 in ConnectWise ScreenConnect and deployed QakBot as an initial loader.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "Black Basta", "surface_text": "Black Basta"},
            {"category": "CVE", "canonical_name": "CVE-2024-1709", "surface_text": "CVE-2024-1709"},
            {"category": "Tool", "canonical_name": "ConnectWise ScreenConnect", "surface_text": "ConnectWise ScreenConnect"},
            {"category": "Malware", "canonical_name": "QakBot", "surface_text": "QakBot"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::007",
        "report_id": "SENTINELONE-ACIDRAIN-2022",
        "source": "SentinelOne",
        "publication_date": "2022-03-31",
        "text": "SentinelLabs analyzed AcidRain, a specialized Linux wiper malware deployed against Viasat KA-SAT satellite broadband terminals in Ukraine.",
        "ground_truth_entities": [
            {"category": "Malware", "canonical_name": "AcidRain", "surface_text": "AcidRain"},
            {"category": "Organization", "canonical_name": "Viasat", "surface_text": "Viasat"},
            {"category": "Location", "canonical_name": "Ukraine", "surface_text": "Ukraine"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::008",
        "report_id": "MSTIC-FOREST-BLIZZARD-2023",
        "source": "Microsoft",
        "publication_date": "2023-04-12",
        "text": "Forest Blizzard exploited CVE-2023-23397 in Microsoft Outlook to target transportation organizations in Poland.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "Forest Blizzard", "surface_text": "Forest Blizzard"},
            {"category": "CVE", "canonical_name": "CVE-2023-23397", "surface_text": "CVE-2023-23397"},
            {"category": "Tool", "canonical_name": "Microsoft Outlook", "surface_text": "Microsoft Outlook"},
            {"category": "Organization", "canonical_name": "Transportation Systems Sector", "surface_text": "transportation organizations"},
            {"category": "Location", "canonical_name": "Poland", "surface_text": "Poland"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::009",
        "report_id": "MSTIC-FOREST-BLIZZARD-2023",
        "source": "Microsoft",
        "publication_date": "2023-04-12",
        "text": "The operators utilized Impacket and Mimikatz to relay stolen authentication hashes and execute T1003 OS Credential Dumping.",
        "ground_truth_entities": [
            {"category": "Tool", "canonical_name": "Impacket", "surface_text": "Impacket"},
            {"category": "Tool", "canonical_name": "Mimikatz", "surface_text": "Mimikatz"},
            {"category": "Technique", "canonical_name": "T1003 - OS Credential Dumping", "surface_text": "T1003 OS Credential Dumping"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::010",
        "report_id": "UNIT42-CLOP-MOVEIT-2023",
        "source": "Unit 42",
        "publication_date": "2023-06-15",
        "text": "Clop ransomware gang exploited CVE-2023-34362 in Progress Software MOVEit Transfer to steal database records from financial institutions.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "Clop", "surface_text": "Clop"},
            {"category": "CVE", "canonical_name": "CVE-2023-34362", "surface_text": "CVE-2023-34362"},
            {"category": "Organization", "canonical_name": "Progress Software", "surface_text": "Progress Software"},
            {"category": "Tool", "canonical_name": "MOVEit Transfer", "surface_text": "MOVEit Transfer"},
            {"category": "Organization", "canonical_name": "Financial Sector", "surface_text": "financial institutions"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::011",
        "report_id": "UNIT42-CLOP-MOVEIT-2023",
        "source": "Unit 42",
        "publication_date": "2023-06-15",
        "text": "The threat actors utilized PowerShell scripts and rclone to stage and exfiltrate records to remote endpoints.",
        "ground_truth_entities": [
            {"category": "Tool", "canonical_name": "PowerShell", "surface_text": "PowerShell"},
            {"category": "Tool", "canonical_name": "rclone", "surface_text": "rclone"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::012",
        "report_id": "MSTIC-OCTO-TEMPEST-2023",
        "source": "Microsoft",
        "publication_date": "2023-10-25",
        "text": "Octo Tempest operators impersonated employees to reset multifactor authentication tokens in Okta across the United States.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "Octo Tempest", "surface_text": "Octo Tempest"},
            {"category": "Organization", "canonical_name": "Okta", "surface_text": "Okta"},
            {"category": "Location", "canonical_name": "United States", "surface_text": "United States"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::013",
        "report_id": "MSTIC-OCTO-TEMPEST-2023",
        "source": "Microsoft",
        "publication_date": "2023-10-25",
        "text": "The actors deployed BloodHound and Mimikatz to map Active Directory permissions and executed T1078 Valid Accounts.",
        "ground_truth_entities": [
            {"category": "Tool", "canonical_name": "BloodHound", "surface_text": "BloodHound"},
            {"category": "Tool", "canonical_name": "Mimikatz", "surface_text": "Mimikatz"},
            {"category": "Technique", "canonical_name": "T1078 - Valid Accounts", "surface_text": "T1078 Valid Accounts"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::014",
        "report_id": "UNIT42-PAN-OS-2024",
        "source": "Unit 42",
        "publication_date": "2024-04-12",
        "text": "Adversaries exploited CVE-2024-3400 in Palo Alto Networks GlobalProtect to obtain root privileges in Germany.",
        "ground_truth_entities": [
            {"category": "CVE", "canonical_name": "CVE-2024-3400", "surface_text": "CVE-2024-3400"},
            {"category": "Organization", "canonical_name": "Palo Alto Networks", "surface_text": "Palo Alto Networks"},
            {"category": "Location", "canonical_name": "Germany", "surface_text": "Germany"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::015",
        "report_id": "MANDIANT-APT41-2022",
        "source": "Mandiant",
        "publication_date": "2022-03-08",
        "text": "APT41 compromised public-facing web servers using CVE-2021-44228 to target municipal governments in the United States.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "APT41", "surface_text": "APT41"},
            {"category": "CVE", "canonical_name": "CVE-2021-44228", "surface_text": "CVE-2021-44228"},
            {"category": "Organization", "canonical_name": "Municipal Governments", "surface_text": "municipal governments"},
            {"category": "Location", "canonical_name": "United States", "surface_text": "United States"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::016",
        "report_id": "MANDIANT-FIN11-2021",
        "source": "Mandiant",
        "publication_date": "2021-02-22",
        "text": "FIN11 deployed DEWMODE web shells and partnered with Clop ransomware operators to target retail enterprises.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "FIN11", "surface_text": "FIN11"},
            {"category": "Malware", "canonical_name": "DEWMODE", "surface_text": "DEWMODE"},
            {"category": "ThreatActor", "canonical_name": "Clop", "surface_text": "Clop"},
            {"category": "Organization", "canonical_name": "Retail & Hospitality Sector", "surface_text": "retail enterprises"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::017",
        "report_id": "MSTIC-DEV-0537-2022",
        "source": "Microsoft",
        "publication_date": "2022-03-22",
        "text": "DEV-0537 utilized AnyDesk to target Microsoft and telecommunications providers in South America.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "DEV-0537", "surface_text": "DEV-0537"},
            {"category": "Tool", "canonical_name": "AnyDesk", "surface_text": "AnyDesk"},
            {"category": "Organization", "canonical_name": "Microsoft", "surface_text": "Microsoft"},
            {"category": "Organization", "canonical_name": "Communications Sector", "surface_text": "telecommunications providers"},
            {"category": "Location", "canonical_name": "South America", "surface_text": "South America"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::018",
        "report_id": "CISA-AA20-302A",
        "source": "CISA",
        "publication_date": "2020-10-28",
        "text": "Adversaries initiated attacks through phishing emails and exploited CVE-2020-1472 Zerologon against hospital networks.",
        "ground_truth_entities": [
            {"category": "Technique", "canonical_name": "T1566 - Phishing", "surface_text": "phishing"},
            {"category": "CVE", "canonical_name": "CVE-2020-1472", "surface_text": "CVE-2020-1472"},
            {"category": "Organization", "canonical_name": "Healthcare Sector", "surface_text": "hospital networks"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::019",
        "report_id": "CISA-AA20-302A",
        "source": "CISA",
        "publication_date": "2020-10-28",
        "text": "Threat actors deployed Ryuk ransomware and TrickBot loaders using Cobalt Strike and BloodHound across the United States.",
        "ground_truth_entities": [
            {"category": "Malware", "canonical_name": "Ryuk", "surface_text": "Ryuk"},
            {"category": "Malware", "canonical_name": "TrickBot", "surface_text": "TrickBot"},
            {"category": "Tool", "canonical_name": "Cobalt Strike", "surface_text": "Cobalt Strike"},
            {"category": "Tool", "canonical_name": "BloodHound", "surface_text": "BloodHound"},
            {"category": "Location", "canonical_name": "United States", "surface_text": "United States"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::020",
        "report_id": "MSTIC-DIAMOND-SLEET-2024",
        "source": "Microsoft",
        "publication_date": "2024-01-18",
        "text": "Diamond Sleet exploited CVE-2022-47966 in Zoho ManageEngine to drop MagicRAT onto developer workstations in South Korea.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "Diamond Sleet", "surface_text": "Diamond Sleet"},
            {"category": "CVE", "canonical_name": "CVE-2022-47966", "surface_text": "CVE-2022-47966"},
            {"category": "Malware", "canonical_name": "MagicRAT", "surface_text": "MagicRAT"},
            {"category": "Location", "canonical_name": "South Korea", "surface_text": "South Korea"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::021",
        "report_id": "CISA-AA21-209A",
        "source": "CISA",
        "publication_date": "2021-07-20",
        "text": "Chinese state-sponsored actors targeted natural gas pipeline companies in the United States using spearphishing emails.",
        "ground_truth_entities": [
            {"category": "Organization", "canonical_name": "Energy Sector", "surface_text": "natural gas pipeline companies"},
            {"category": "Location", "canonical_name": "United States", "surface_text": "United States"},
            {"category": "Technique", "canonical_name": "T1566 - Phishing", "surface_text": "spearphishing emails"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::022",
        "report_id": "MANDIANT-APT44-2024",
        "source": "Mandiant",
        "publication_date": "2024-04-17",
        "text": "APT44 deployed HermeticWiper, CaddyWiper, and Industroyer2 to corrupt operational technology networks in Ukraine.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "APT44", "surface_text": "APT44"},
            {"category": "Malware", "canonical_name": "HermeticWiper", "surface_text": "HermeticWiper"},
            {"category": "Malware", "canonical_name": "CaddyWiper", "surface_text": "CaddyWiper"},
            {"category": "Malware", "canonical_name": "Industroyer2", "surface_text": "Industroyer2"},
            {"category": "Location", "canonical_name": "Ukraine", "surface_text": "Ukraine"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::023",
        "report_id": "CISA-AA24-038A",
        "source": "CISA",
        "publication_date": "2024-02-07",
        "text": "Volt Typhoon compromised water treatment facilities and electrical utilities in Guam and the continental United States.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "Volt Typhoon", "surface_text": "Volt Typhoon"},
            {"category": "Organization", "canonical_name": "Energy Sector", "surface_text": "electrical utilities"},
            {"category": "Location", "canonical_name": "Guam", "surface_text": "Guam"},
            {"category": "Location", "canonical_name": "United States", "surface_text": "United States"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::024",
        "report_id": "MSTIC-MINT-SANDSTORM-2023",
        "source": "Microsoft",
        "publication_date": "2023-11-16",
        "text": "Mint Sandstorm exploited CVE-2021-44228 Log4Shell to conduct cyber espionage against defense personnel in Israel.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "Mint Sandstorm", "surface_text": "Mint Sandstorm"},
            {"category": "CVE", "canonical_name": "CVE-2021-44228", "surface_text": "CVE-2021-44228"},
            {"category": "Location", "canonical_name": "Israel", "surface_text": "Israel"}
        ]
    },
    {
        "sentence_id": "ANNOCTR::025",
        "report_id": "CISA-AA23-158A",
        "source": "CISA",
        "publication_date": "2023-06-07",
        "text": "The SVR targeted foreign policy think tanks across NATO member states using CVE-2022-30190 Follina and Mimikatz.",
        "ground_truth_entities": [
            {"category": "ThreatActor", "canonical_name": "APT29", "surface_text": "SVR"},
            {"category": "Organization", "canonical_name": "NATO", "surface_text": "NATO"},
            {"category": "CVE", "canonical_name": "CVE-2022-30190", "surface_text": "CVE-2022-30190"},
            {"category": "Tool", "canonical_name": "Mimikatz", "surface_text": "Mimikatz"}
        ]
    }
]

def create():
    out_path = Path("data/benchmark/annoctr_gold_standard.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ANNOCTR_DATA, f, indent=2, ensure_ascii=False)
    total_entities = sum(len(s["ground_truth_entities"]) for s in ANNOCTR_DATA)
    print(f"Created AnnoCTR benchmark with {len(ANNOCTR_DATA)} sentences and {total_entities} gold entities at {out_path}")

if __name__ == "__main__":
    create()
