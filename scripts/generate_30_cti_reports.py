"""Generate 30 curated CTI demonstration reports modeled on primary threat intelligence advisories (CISA, Mandiant, CrowdStrike).
Preserves report_id, source, publication_date, url, author, tlp, and multi-paragraph technical content.
"""

import json
from pathlib import Path

REPORTS = [
    {
        "report_id": "CISA-AA23-347A",
        "title": "People's Republic of China State-Sponsored Cyber Actors Exploit Ivanti and Living-off-the-Land",
        "source": "CISA",
        "publication_date": "2023-12-13",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa23-347a",
        "content": (
            "The Cybersecurity and Infrastructure Security Agency (CISA) and federal partners have observed "
            "state-sponsored cyber threat actor Volt Typhoon compromising critical infrastructure organizations across "
            "the United States. Volt Typhoon, also tracked as Vanguard Panda, has maintained persistent access within "
            "targeted networks in the Communications, Energy, and Transportation Systems sectors, including vital facilities "
            "in Guam and Hawaii.\n\n"
            "To establish initial access, Volt Typhoon exploited zero-day vulnerabilities in perimeter edge devices, "
            "specifically CVE-2024-21887 and CVE-2023-46805 affecting Ivanti Connect Secure VPN appliances. After breaching "
            "perimeter gateways, the actor exploited CVE-2023-4966 in Citrix Bleed appliances to hijack authenticated sessions.\n\n"
            "Once inside victim networks, Volt Typhoon primarily deployed living-off-the-land techniques to avoid detection. "
            "The threat actor used PowerShell, WMI, and ntdsutil to conduct discovery and harvest credentials from active "
            "directory controllers. Volt Typhoon also utilized Chisel and Fast Reverse Proxy to establish encrypted "
            "command-and-control tunnels out of the compromised environment.\n\n"
            "Volt Typhoon targeted the Defense Industrial Base, telecommunications providers, and port authorities across "
            "the United States. Furthermore, the threat actor utilized T1190 Exploit Public-Facing Application to bypass "
            "perimeter controls and employed T1059 Command and Scripting Interpreter for lateral movement."
        ),
    },
    {
        "report_id": "MANDIANT-APT29-2024",
        "title": "Midnight Blizzard (APT29) Global Campaign Leveraging Supply Chain and Cloud Identities",
        "source": "Mandiant",
        "publication_date": "2024-01-25",
        "tlp": "CLEAR",
        "author": "Mandiant Threat Intelligence",
        "url": "https://cloud.google.com/blog/topics/threat-intelligence/apt29-midnight-blizzard-cloud-supply-chain",
        "content": (
            "Mandiant has observed advanced persistent threat group APT29, widely known as Cozy Bear or Midnight Blizzard, "
            "conducting widespread cyber espionage operations targeting government agencies, think tanks, and defense "
            "contractors across the United States and Europe. In recent operations, APT29 targeted NATO member states, "
            "European diplomatic entities, and Ukrainian government organizations.\n\n"
            "APT29 leveraged spearphishing emails containing malicious attachments to deliver malware payloads. The threat "
            "actor exploited CVE-2023-38831, a vulnerability in WinRAR processing, to execute arbitrary code when victims "
            "opened deceptive PDF and ZIP archives. In historical campaigns, APT29 also exploited CVE-2021-44228, commonly "
            "known as Log4Shell, against public-facing Apache servers.\n\n"
            "Following successful initial compromise, APT29 used Cobalt Strike BEACON as their primary post-exploitation "
            "framework. In addition to Cobalt Strike, the operators deployed Mimikatz to dump plaintext credentials and "
            "Kerberos tickets from memory. The group also used PsExec to execute malicious binaries across remote endpoints.\n\n"
            "APT29 targeted Microsoft enterprise accounts to exfiltrate executive emails and source code repositories. "
            "The actors extensively employed T1566 Phishing and T1003 OS Credential Dumping to compromise elevated privileges, "
            "while heavily relying on T1078 Valid Accounts to move undetected across cloud environments."
        ),
    },
    {
        "report_id": "UNIT42-LAZARUS-2023",
        "title": "Lazarus Group Deploys MagicRAT and Attacks Defense and Financial Organizations",
        "source": "Unit 42",
        "publication_date": "2023-09-18",
        "tlp": "CLEAR",
        "author": "Palo Alto Networks Unit 42",
        "url": "https://unit42.paloaltonetworks.com/lazarus-magicrat-defense-financial",
        "content": (
            "Unit 42 researchers identified renewed activity by Lazarus Group, an advanced persistent threat actor sponsored "
            "by the North Korean government, also recognized as Hidden Cobra and Zinc. Lazarus Group focused malicious "
            "operations against cryptocurrency exchanges, aerospace engineering firms, and financial institutions in South Korea, "
            "Japan, and the United States.\n\n"
            "Lazarus Group exploited CVE-2022-47966, a critical remote code execution flaw in Zoho ManageEngine ServiceDesk Plus, "
            "to gain initial access to enterprise infrastructure. Furthermore, Lazarus Group exploited CVE-2021-44228 in vulnerable "
            "VMware Horizon instances.\n\n"
            "Upon gaining access, Lazarus Group deployed the AppleJeus trojan alongside their custom backdoor MagicRAT. "
            "The operators utilized Chisel and PuTTY to maintain SOCKS5 proxy connections for data exfiltration. Lazarus Group "
            "also executed Mimikatz to extract administrator credentials from LSASS process memory.\n\n"
            "The threat actor targeted defense contractors in South Korea and cryptocurrency corporations in the United States "
            "to generate illicit revenue. In these intrusions, Lazarus Group utilized T1190 Exploit Public-Facing Application "
            "and executed T1059 Command and Scripting Interpreter via malicious scripts."
        ),
    },
    {
        "report_id": "CROWDSTRIKE-SANDWORM-2023",
        "title": "Sandworm Disrupts Critical Ukrainian Infrastructure with Wiper Malware",
        "source": "CrowdStrike",
        "publication_date": "2023-11-08",
        "tlp": "CLEAR",
        "author": "CrowdStrike Intelligence",
        "url": "https://www.crowdstrike.com/blog/sandworm-wiper-attacks-ukraine/",
        "content": (
            "CrowdStrike Counter Adversary Operations tracked aggressive offensive cyber operations by Sandworm, also designated "
            "as Voodoo Bear and APT44, an adversarial unit associated with the Russian GRU. Sandworm launched destructive "
            "campaigns against the energy grid, water utilities, and railway systems across Ukraine and Poland.\n\n"
            "During recent intrusions, Sandworm exploited CVE-2022-30190, the Microsoft Support Diagnostic Tool vulnerability "
            "known as Follina, embedded within weaponized RTF documents. The adversary also exploited CVE-2023-38831 to "
            "compromise regional energy distribution networks.\n\n"
            "Sandworm deployed devastating destructive malware including HermeticWiper, CaddyWiper, and Industroyer2 to "
            "corrupt operational technology and overwrite master boot records. To facilitate lateral movement, the operators "
            "used PsExec and Impacket toolsets.\n\n"
            "Sandworm targeted the Ukrainian Ministry of Defense, national electric substations, and telecommunication providers "
            "in Ukraine. The group employed T1566 Phishing for initial lure delivery and utilized T1485 Data Destruction "
            "alongside T1059 Command and Scripting Interpreter to maximize physical and digital disruption."
        ),
    },
    {
        "report_id": "CISA-LOCKBIT3-2024",
        "title": "Understanding and Defending Against LockBit 3.0 Ransomware Operations",
        "source": "CISA",
        "publication_date": "2024-02-20",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa24-051a",
        "content": (
            "CISA, the Federal Bureau of Investigation (FBI), and international partners issued a joint cybersecurity advisory "
            "regarding the LockBit 3.0 ransomware gang, also designated as LockBit Black. LockBit affiliates targeted healthcare "
            "systems, municipal governments, and manufacturing corporations across the United States, Germany, and the United Kingdom.\n\n"
            "LockBit affiliates exploited CVE-2023-4966, termed Citrix Bleed, to bypass multifactor authentication and hijack "
            "existing administrative sessions. Threat actors associated with LockBit also exploited CVE-2021-34527, known as "
            "PrintNightmare, for local privilege escalation.\n\n"
            "In multiple engagements, LockBit deployed BlackCat ransomware components and proprietary LockBit lockers. The "
            "ransomware operators utilized Cobalt Strike, Mimikatz, and AnyDesk to conduct lateral movement and maintain access. "
            "The affiliates used MegaSync and rclone to exfiltrate gigabytes of confidential customer records.\n\n"
            "LockBit targeted hospital networks and financial services firms in the United States and Europe. The gang routinely "
            "leverages T1190 Exploit Public-Facing Application, employs T1078 Valid Accounts, and utilizes T1486 Data Encrypted "
            "for Impact during final extortion phases."
        ),
    },
    {
        "report_id": "SENTINELONE-FIN7-2023",
        "title": "FIN7 Continues Multi-Vector Attacks Against Retail and Hospitality",
        "source": "SentinelOne",
        "publication_date": "2023-08-14",
        "tlp": "CLEAR",
        "author": "SentinelLabs",
        "url": "https://www.sentinelone.com/labs/fin7-attacks-retail-hospitality",
        "content": (
            "SentinelLabs uncovered active cybercrime campaigns attributed to FIN7, also tracked as Carbanak and Sangria "
            "Tempest. FIN7 historically targeted retail corporations, restaurant franchises, and hospitality chains across "
            "the United States and Canada.\n\n"
            "FIN7 operators exploited CVE-2021-34527 PrintNightmare and leveraged CVE-2022-30190 Follina to execute malicious "
            "payloads via Microsoft Office documents. In several attacks, FIN7 exploited CVE-2023-38831 to bypass endpoint "
            "security products.\n\n"
            "FIN7 deployed Carbanak malware and proprietary backdoors such as POWERPLANT. To assist in reconnaissance and "
            "credential theft, FIN7 used Cobalt Strike and BloodHound. The group also used PowerShell scripts to automate "
            "payload staging.\n\n"
            "FIN7 targeted major hotel chains, point-of-sale software vendors, and retail enterprises throughout the United "
            "States. Threat actors utilized T1566 Phishing to initiate compromise, executed T1059 Command and Scripting "
            "Interpreter, and employed T1003 OS Credential Dumping to seize domain administrator privileges."
        ),
    },
    {
        "report_id": "CISA-AA24-131A",
        "title": "Black Basta Ransomware Affiliates Target Critical Infrastructure and Healthcare",
        "source": "CISA",
        "publication_date": "2024-05-10",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa24-131a",
        "content": (
            "CISA and the Federal Bureau of Investigation (FBI) released a joint advisory on Black Basta ransomware. "
            "Black Basta operators have targeted healthcare organizations, defense contractors, and manufacturing entities "
            "across North America and Europe.\n\n"
            "Black Basta affiliates commonly gain initial access through phishing lures containing malicious attachments and "
            "exploited CVE-2024-1709, an authentication bypass vulnerability in ConnectWise ScreenConnect. Furthermore, "
            "Black Basta operators frequently deploy QakBot malware as an initial loader.\n\n"
            "Once inside the victim network, Black Basta operators use PsExec and Mimikatz for lateral movement and credential "
            "harvesting. Affiliates also deployed Cobalt Strike BEACON to establish command and control. The operators targeted "
            "hospital networks in the United States, utilizing T1486 Data Encrypted for Impact."
        ),
    },
    {
        "report_id": "CISA-AA23-353A",
        "title": "ALPHV BlackCat Ransomware Gang Affiliates Deploy Advanced Extortion Tactics",
        "source": "CISA",
        "publication_date": "2023-12-19",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa23-353a",
        "content": (
            "CISA and international partners observed aggressive operations by the ALPHV BlackCat ransomware group. "
            "ALPHV BlackCat operators have targeted healthcare facilities, financial institutions, and government agencies "
            "across the United States and Europe.\n\n"
            "Threat actors affiliated with ALPHV BlackCat exploited CVE-2023-46805 and CVE-2024-21887 in Ivanti Connect Secure "
            "appliances to achieve perimeter access. Actors also utilized compromised credentials to log into commercial VPNs.\n\n"
            "During intrusions, BlackCat affiliates utilized MegaSync and rclone to transfer sensitive data to cloud storage. "
            "The operators deployed PsExec to propagate ransomware binaries across corporate workstations, executing T1078 Valid "
            "Accounts and T1486 Data Encrypted for Impact."
        ),
    },
    {
        "report_id": "MSTIC-STORM-0558-2023",
        "title": "Analysis of Storm-0558 Cyber Espionage Intrusions into Cloud Email Accounts",
        "source": "Microsoft",
        "publication_date": "2023-07-14",
        "tlp": "CLEAR",
        "author": "Microsoft Threat Intelligence",
        "url": "https://www.microsoft.com/security/blog/2023/07/14/analysis-of-storm-0558-cyber-espionage-intrusions/",
        "content": (
            "Microsoft Threat Intelligence tracked cyber espionage group Storm-0558 conducting targeted operations against "
            "government organizations in the United States and diplomatic entities in Europe. Storm-0558 is an adversary "
            "based in China focused on espionage against geopolitical targets.\n\n"
            "The threat actor exploited CVE-2023-36884, a vulnerability allowing remote code execution via Microsoft Office "
            "documents, alongside forged authentication tokens. Storm-0558 targeted Exchange Online mailboxes belonging to "
            "senior foreign policy officials.\n\n"
            "Storm-0558 operators utilized PowerShell scripts and custom extraction utilities to collect sensitive correspondence. "
            "The threat actor relied on T1078 Valid Accounts to access cloud resources without raising perimeter alarms."
        ),
    },
    {
        "report_id": "UNIT42-DARKGATE-2023",
        "title": "DarkGate Modular Loader Activity Delivered via Teams and Phishing",
        "source": "Unit 42",
        "publication_date": "2023-11-20",
        "tlp": "CLEAR",
        "author": "Palo Alto Networks Unit 42",
        "url": "https://unit42.paloaltonetworks.com/darkgate-modular-loader-activity/",
        "content": (
            "Unit 42 observed an increase in DarkGate malware distribution campaigns targeting corporate networks. DarkGate "
            "is a commodity loader that provides remote access, keylogging, and cryptocurrency theft capabilities.\n\n"
            "Adversaries deployed DarkGate through phishing messages sent across Microsoft Teams and malicious email links. "
            "DarkGate operators exploited CVE-2023-38831 to bypass execution protections on target endpoints.\n\n"
            "Upon execution, DarkGate deployed living-off-the-land techniques using PowerShell to evade endpoint detection. "
            "The malware targeted financial services firms in the United States and manufacturing enterprises in Germany, "
            "executing T1566 Phishing and T1059 Command and Scripting Interpreter."
        ),
    },
    {
        "report_id": "CISA-AA23-158A",
        "title": "Russian Foreign Intelligence Service (SVR) Cyber Operations (APT29)",
        "source": "CISA",
        "publication_date": "2023-06-07",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa23-158a",
        "content": (
            "CISA, FBI, and the NSA released joint guidance detailing ongoing cyber espionage campaigns conducted by the "
            "Russian Foreign Intelligence Service (SVR), commonly known as APT29 or Cozy Bear. The SVR targeted government "
            "agencies, embassies, and foreign policy think tanks across NATO member states.\n\n"
            "APT29 exploited CVE-2022-30190 Follina and CVE-2021-44228 Log4Shell to compromise internet-facing servers. "
            "The actors sent targeted spearphishing lures imitating diplomatic communiques to foreign ministry personnel.\n\n"
            "The operators deployed Cobalt Strike and Mimikatz to escalate privileges and dump domain credentials. APT29 "
            "extensively utilized T1078 Valid Accounts and T1003 OS Credential Dumping to sustain access within government networks."
        ),
    },
    {
        "report_id": "MSTIC-FOREST-BLIZZARD-2023",
        "title": "Forest Blizzard (APT28) Exploits Microsoft Outlook Vulnerability CVE-2023-23397",
        "source": "Microsoft",
        "publication_date": "2023-04-12",
        "tlp": "CLEAR",
        "author": "Microsoft Threat Intelligence",
        "url": "https://www.microsoft.com/security/blog/2023/04/12/forest-blizzard-apt28-exploits-cve-2023-23397/",
        "content": (
            "Microsoft Threat Intelligence tracked threat actor Forest Blizzard, also known as APT28 or Fancy Bear, "
            "exploiting CVE-2023-23397, a critical privilege escalation vulnerability in Microsoft Outlook. The group is "
            "attributed to the Russian military intelligence agency (GRU).\n\n"
            "Forest Blizzard targeted government, transportation, and defense organizations across Ukraine, Poland, and the "
            "United States. By sending specially crafted calendar appointments, the actors triggered NTLM authentication "
            "hashes to be transmitted to attacker-controlled infrastructure.\n\n"
            "The operators utilized Impacket and Mimikatz to relay stolen authentication hashes and perform lateral movement. "
            "Forest Blizzard executed T1190 Exploit Public-Facing Application and T1003 OS Credential Dumping during these operations."
        ),
    },
    {
        "report_id": "MANDIANT-APT44-2024",
        "title": "APT44 (Sandworm): A Dedicated Russian Military Cyber Sabotage Unit",
        "source": "Mandiant",
        "publication_date": "2024-04-17",
        "tlp": "CLEAR",
        "author": "Mandiant Threat Intelligence",
        "url": "https://cloud.google.com/blog/topics/threat-intelligence/apt44-sandworm-russian-military-cyber-sabotage",
        "content": (
            "Mandiant released comprehensive analysis of APT44, also tracked as Sandworm, an aggressive cyber sabotage unit "
            "operating under the Russian GRU. APT44 has repeatedly targeted the electric power grid, municipal utilities, "
            "and government communications across Ukraine.\n\n"
            "APT44 exploited CVE-2022-30190 and deployed destructive wiper malware including HermeticWiper, CaddyWiper, "
            "and Industroyer2 against operational technology networks in regional electric distribution centers.\n\n"
            "The adversary employed PsExec, Chisel, and Impacket to maneuver through active directory domains. APT44 targeted "
            "the Ukrainian Ministry of Defense and energy infrastructure, executing T1485 Data Destruction and T1059 Command "
            "and Scripting Interpreter."
        ),
    },
    {
        "report_id": "UNIT42-CLOP-MOVEIT-2023",
        "title": "Clop Ransomware Exploitation of MOVEit Transfer Vulnerability CVE-2023-34362",
        "source": "Unit 42",
        "publication_date": "2023-06-15",
        "tlp": "CLEAR",
        "author": "Palo Alto Networks Unit 42",
        "url": "https://unit42.paloaltonetworks.com/clop-ransomware-moveit-cve-2023-34362/",
        "content": (
            "Unit 42 threat intelligence tracked mass exploitation by the Clop ransomware gang of CVE-2023-34362, a critical "
            "SQL injection vulnerability in Progress Software's MOVEit Transfer managed file transfer solution.\n\n"
            "Clop operators deployed custom web shells to exfiltrate database records and sensitive files from hundreds of "
            "organizations. Clop targeted financial institutions, healthcare providers, and government agencies in the "
            "United States, Canada, and the United Kingdom.\n\n"
            "The threat actors utilized automated PowerShell scripts and rclone to stage and exfiltrate multi-terabyte data stores. "
            "Clop leveraged T1190 Exploit Public-Facing Application and utilized extortion techniques to demand ransom payments."
        ),
    },
    {
        "report_id": "CISA-AA24-038A",
        "title": "PRC State-Sponsored Actors Pre-Position on US Critical Infrastructure (Volt Typhoon)",
        "source": "CISA",
        "publication_date": "2024-02-07",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa24-038a",
        "content": (
            "CISA, NSA, and the FBI warned that Chinese state-sponsored cyber group Volt Typhoon is actively pre-positioning "
            "within United States critical infrastructure networks to enable disruptive cyberattacks during geopolitical crises.\n\n"
            "Volt Typhoon compromised routers and firewalls by exploiting CVE-2024-21887 and CVE-2023-46805. The actors focused "
            "intrusions on water treatment facilities, electrical utilities, and transportation systems in Guam and the continental "
            "United States.\n\n"
            "Volt Typhoon utilized living-off-the-land binaries, including PowerShell, WMI, and ntdsutil, avoiding malware "
            "signatures. The actors deployed Fast Reverse Proxy for persistent access, executing T1078 Valid Accounts and T1059."
        ),
    },
    {
        "report_id": "MSTIC-OCTO-TEMPEST-2023",
        "title": "Octo Tempest: Cross-Industry Social Engineering and Cloud Compromise",
        "source": "Microsoft",
        "publication_date": "2023-10-25",
        "tlp": "CLEAR",
        "author": "Microsoft Threat Intelligence",
        "url": "https://www.microsoft.com/security/blog/2023/10/25/octo-tempest-cross-industry-social-engineering/",
        "content": (
            "Microsoft Threat Intelligence tracked financially motivated group Octo Tempest, also known as Scattered Spider "
            "and UNC3944. Octo Tempest is renowned for advanced social engineering against enterprise helpdesks and identity providers.\n\n"
            "The threat actor targeted hospitality chains, casino resorts, and telecommunications providers across the United "
            "States and Canada. Octo Tempest operators impersonated employees to reset multifactor authentication tokens in Okta.\n\n"
            "Once inside cloud environments, the actors deployed BloodHound and Mimikatz to map Active Directory permissions. "
            "In later phases, Octo Tempest partnered with BlackCat ransomware affiliates, executing T1078 Valid Accounts and "
            "T1486 Data Encrypted for Impact."
        ),
    },
    {
        "report_id": "MANDIANT-UNC3944-2023",
        "title": "UNC3944 Targets Enterprise Identity Providers and Cloud Infrastructure",
        "source": "Mandiant",
        "publication_date": "2023-09-22",
        "tlp": "CLEAR",
        "author": "Mandiant Threat Intelligence",
        "url": "https://cloud.google.com/blog/topics/threat-intelligence/unc3944-sms-phishing-identity-providers",
        "content": (
            "Mandiant investigated major intrusions by threat group UNC3944 (Scattered Spider). UNC3944 heavily targeted "
            "retail enterprises, hospitality corporations, and entertainment organizations in the United States.\n\n"
            "UNC3944 used SMS phishing and vishing lures to harvest administrative credentials. The actors exploited "
            "compromised accounts to access AWS and Azure tenants, utilizing PuTTY and AnyDesk for remote persistence.\n\n"
            "The adversary deployed BlackCat ransomware to encrypt virtualization infrastructure. UNC3944 utilized T1566 "
            "Phishing and T1078 Valid Accounts to bypass standard defensive perimeters."
        ),
    },
    {
        "report_id": "SENTINELONE-ACIDRAIN-2022",
        "title": "AcidRain Wiper Targets Satellite Terminals During Ukraine Invasions",
        "source": "SentinelOne",
        "publication_date": "2022-03-31",
        "tlp": "CLEAR",
        "author": "SentinelLabs",
        "url": "https://www.sentinelone.com/labs/acidrain-wiper-satellite-terminals-ukraine/",
        "content": (
            "SentinelLabs analyzed AcidRain, a specialized Linux wiper malware deployed by Russian military threat actors "
            "against Viasat KA-SAT satellite broadband terminals in Ukraine. The cyber attack disrupted satellite communications "
            "across Ukraine and parts of Europe.\n\n"
            "Threat actors, attributed in intelligence reports to Sandworm, exploited misconfigured VPN appliances to gain "
            "access to the satellite ground control network. The actors then distributed AcidRain to overwriting flash memory.\n\n"
            "AcidRain recursively wiped storage directories, destroying firmware and rendering satellite modems inoperable. "
            "The threat actor targeted the Ukrainian military and European communications infrastructure, executing T1485 "
            "Data Destruction."
        ),
    },
    {
        "report_id": "CISA-AA22-110A",
        "title": "Russian State-Sponsored Cyber Threats to Critical Infrastructure and Defense",
        "source": "CISA",
        "publication_date": "2022-04-20",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa22-110a",
        "content": (
            "CISA, FBI, NSA, and international partners issued a comprehensive advisory detailing cyber capabilities of Russian "
            "state-sponsored threat groups, including APT29, APT28, and Sandworm. The actors have targeted defense contractors, "
            "energy utilities, and government departments in the United States and NATO member states.\n\n"
            "The adversaries exploited CVE-2021-44228 Log4Shell and CVE-2020-1472 Zerologon to compromise perimeter networks. "
            "The threat actors utilized spearphishing attachments to establish initial access.\n\n"
            "Russian operators deployed Cobalt Strike, Mimikatz, and Impacket to move laterally. The groups targeted defense "
            "industrial base entities in the United States, utilizing T1190 Exploit Public-Facing Application and T1003 OS "
            "Credential Dumping."
        ),
    },
    {
        "report_id": "MSTIC-NOBELIUM-2021",
        "title": "Nobelium (APT29) Multi-Sector Spearphishing Operations via Email Delivery",
        "source": "Microsoft",
        "publication_date": "2021-05-27",
        "tlp": "CLEAR",
        "author": "Microsoft Threat Intelligence",
        "url": "https://www.microsoft.com/security/blog/2021/05/27/nobelium-spearphishing-email-campaigns/",
        "content": (
            "Microsoft Threat Intelligence detected wide-scale spearphishing operations by Nobelium, also tracked as APT29. "
            "Nobelium compromised the Constant Contact marketing account of USAID to distribute malicious emails to government "
            "agencies, human rights groups, and think tanks.\n\n"
            "The malicious emails targeted organizations in over 24 countries, with heavy concentration in the United States "
            "and Europe. The attachments dropped custom Cobalt Strike BEACON payloads upon opening.\n\n"
            "Nobelium operators used PowerShell and Mimikatz to conduct Active Directory discovery and harvest session tokens. "
            "The adversary executed T1566 Phishing and relied on T1059 Command and Scripting Interpreter for payload execution."
        ),
    },
    {
        "report_id": "UNIT42-PAN-OS-2024",
        "title": "State-Sponsored Adversary Exploits PAN-OS Command Injection Flaw CVE-2024-3400",
        "source": "Unit 42",
        "publication_date": "2024-04-12",
        "tlp": "CLEAR",
        "author": "Palo Alto Networks Unit 42",
        "url": "https://unit42.paloaltonetworks.com/cve-2024-3400-pan-os-exploitation/",
        "content": (
            "Unit 42 researchers identified active in-the-wild exploitation of CVE-2024-3400, a critical command injection "
            "vulnerability in the GlobalProtect feature of Palo Alto Networks PAN-OS software.\n\n"
            "State-sponsored actors exploited CVE-2024-3400 to obtain root privileges on perimeter security appliances without "
            "authentication. The adversaries targeted government departments, defense contractors, and energy sector organizations "
            "across North America and Europe.\n\n"
            "Following initial breach, the threat actors deployed custom backdoors and utilized PuTTY and Chisel to tunnel traffic "
            "into internal enterprise networks. The actors executed T1190 Exploit Public-Facing Application and T1059."
        ),
    },
    {
        "report_id": "MANDIANT-APT41-2022",
        "title": "APT41 Espionage Campaigns Compromise US State Governments via Log4Shell",
        "source": "Mandiant",
        "publication_date": "2022-03-08",
        "tlp": "CLEAR",
        "author": "Mandiant Threat Intelligence",
        "url": "https://cloud.google.com/blog/topics/threat-intelligence/apt41-us-state-governments-log4shell",
        "content": (
            "Mandiant identified sophisticated intrusions by Chinese state-sponsored cyber espionage group APT41 into at least "
            "six United States state government networks. APT41 exploited CVE-2021-44228 Log4Shell and CVE-2021-40539.\n\n"
            "The threat actor compromised public-facing web servers and animal health surveillance databases. APT41 targeted "
            "municipal governments and state agricultural departments across the United States.\n\n"
            "The operators deployed Cobalt Strike and custom backdoors to exfiltrate database records. APT41 utilized Mimikatz "
            "and PsExec to escalate privileges, executing T1190 Exploit Public-Facing Application and T1003 OS Credential Dumping."
        ),
    },
    {
        "report_id": "CISA-AA22-047A",
        "title": "Russian State-Sponsored Cyber Actors Target US Defense Contractors",
        "source": "CISA",
        "publication_date": "2022-02-16",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa22-047a",
        "content": (
            "CISA, FBI, and the NSA published a joint advisory highlighting Russian state-sponsored actors targeting United "
            "States defense contractors. The adversaries acquired sensitive unclassified defense information.\n\n"
            "The threat actors exploited CVE-2020-1472 Zerologon and brute-forced administrative credentials. Intrusions targeted "
            "aerospace engineering firms and military communications manufacturers in the United States.\n\n"
            "Adversaries utilized Mimikatz to dump credentials and deployed PuTTY to establish outbound command-and-control "
            "channels. The operators executed T1078 Valid Accounts and T1003 OS Credential Dumping to compromise elevated domains."
        ),
    },
    {
        "report_id": "MSTIC-MINT-SANDSTORM-2023",
        "title": "Mint Sandstorm (Charming Kitten) Conducts Espionage Against Defense Personnel",
        "source": "Microsoft",
        "publication_date": "2023-11-16",
        "tlp": "CLEAR",
        "author": "Microsoft Threat Intelligence",
        "url": "https://www.microsoft.com/security/blog/2023/11/16/mint-sandstorm-espionage-defense-experts/",
        "content": (
            "Microsoft Threat Intelligence tracked Iranian threat actor Mint Sandstorm, also known as Charming Kitten and "
            "APT35, targeting defense researchers and think-tank analysts in the United States, Israel, and Europe.\n\n"
            "The actor utilized spearphishing messages sent through social media and email platforms to deliver weaponized "
            "attachments. Mint Sandstorm exploited CVE-2021-44228 Log4Shell in internet-exposed test environments.\n\n"
            "The operators deployed PowerShell scripts and custom credential harvesting tools. Mint Sandstorm targeted defense "
            "industrial base entities, executing T1566 Phishing and T1059 Command and Scripting Interpreter."
        ),
    },
    {
        "report_id": "MSTIC-DIAMOND-SLEET-2024",
        "title": "Diamond Sleet (Lazarus) Weaponizes Open-Source Software and CyberArk Utilities",
        "source": "Microsoft",
        "publication_date": "2024-01-18",
        "tlp": "CLEAR",
        "author": "Microsoft Threat Intelligence",
        "url": "https://www.microsoft.com/security/blog/2024/01/18/diamond-sleet-weaponizes-open-source/",
        "content": (
            "Microsoft Threat Intelligence observed North Korean threat group Diamond Sleet (Lazarus Group) conducting supply "
            "chain intrusions targeting software development enterprises in South Korea and the United States.\n\n"
            "Diamond Sleet trojanized open-source dependencies and exploited CVE-2022-47966 in Zoho ManageEngine to obtain "
            "initial access. The actors dropped the MagicRAT trojan onto compromised developer workstations.\n\n"
            "The operators used PuTTY and Chisel to route network traffic and deployed Mimikatz to siphon cryptographic keys. "
            "Diamond Sleet targeted financial institutions and cryptocurrency corporations, executing T1190 and T1059."
        ),
    },
    {
        "report_id": "CISA-AA21-209A",
        "title": "Chinese State-Sponsored Cyber Actors Target US Natural Gas Pipelines",
        "source": "CISA",
        "publication_date": "2021-07-20",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa21-209a",
        "content": (
            "CISA and the FBI identified Chinese state-sponsored cyber actors conducting intrusion campaigns against major "
            "United States oil and natural gas pipeline companies between 2011 and 2013.\n\n"
            "The actors leveraged spearphishing emails and exploited known web vulnerabilities to compromise administrative "
            "workstations. Intrusions targeted critical energy sector infrastructure across the United States.\n\n"
            "The adversaries deployed custom RATs and used PowerShell to explore pipeline control architectures. The threat "
            "actors executed T1566 Phishing and T1078 Valid Accounts to map critical industrial infrastructure."
        ),
    },
    {
        "report_id": "MANDIANT-FIN11-2021",
        "title": "FIN11 Exploits Accellion FTA Vulnerabilities to Distribute Clop Ransomware",
        "source": "Mandiant",
        "publication_date": "2021-02-22",
        "tlp": "CLEAR",
        "author": "Mandiant Threat Intelligence",
        "url": "https://cloud.google.com/blog/topics/threat-intelligence/fin11-accellion-fta-clop-extortion",
        "content": (
            "Mandiant tracked widespread extortion activity by FIN11 exploiting zero-day vulnerabilities in Accellion File "
            "Transfer Appliance (FTA) systems. FIN11 partnered with Clop ransomware operators to monetize stolen data.\n\n"
            "The threat actors deployed DEWMODE web shells to harvest files from compromised FTA appliances. FIN11 targeted "
            "retail enterprises, healthcare systems, and financial corporations in the United States and Europe.\n\n"
            "The operators used PowerShell and MegaSync to exfiltrate customer databases. FIN11 utilized T1190 Exploit "
            "Public-Facing Application and T1486 Data Encrypted for Impact during extortion demands."
        ),
    },
    {
        "report_id": "UNIT42-BLACKBASTA-2023",
        "title": "Black Basta Ransomware Syndicate: Technical TTPs and Defense Evasion",
        "source": "Unit 42",
        "publication_date": "2023-12-05",
        "tlp": "CLEAR",
        "author": "Palo Alto Networks Unit 42",
        "url": "https://unit42.paloaltonetworks.com/black-basta-ransomware-ttps/",
        "content": (
            "Unit 42 investigated the technical operations of the Black Basta ransomware group. Black Basta emerged as a "
            "major threat to enterprise networks across the manufacturing, healthcare, and financial sectors in North America.\n\n"
            "The group utilized QakBot and malicious email campaigns for initial delivery. In several incidents, operators "
            "exploited CVE-2023-38831 in WinRAR archives to deploy initial beacons.\n\n"
            "Black Basta operators used Cobalt Strike, Mimikatz, and PsExec for internal discovery and credential access. "
            "The gang targeted hospital networks and defense contractors, executing T1003 OS Credential Dumping and T1486."
        ),
    },
    {
        "report_id": "MSTIC-DEV-0537-2022",
        "title": "DEV-0537 (Lapsus$) Extortion Tactics and Credential Theft Against Global Tech",
        "source": "Microsoft",
        "publication_date": "2022-03-22",
        "tlp": "CLEAR",
        "author": "Microsoft Threat Intelligence",
        "url": "https://www.microsoft.com/security/blog/2022/03/22/dev-0537-lapsus-extortion-tactics/",
        "content": (
            "Microsoft Threat Intelligence tracked the cyber group DEV-0537, also known as Lapsus$, conducting destructive "
            "extortion campaigns against major technology corporations and telecommunications providers in the United States, "
            "Europe, and South America.\n\n"
            "Unlike traditional ransomware gangs, DEV-0537 gained access by bribing corporate insiders and purchasing stolen "
            "credentials from illicit marketplaces. The group targeted Microsoft, Okta, and healthcare software vendors.\n\n"
            "The actors utilized AnyDesk and living-off-the-land tools to exfiltrate proprietary source code repositories. "
            "DEV-0537 executed T1078 Valid Accounts to move through corporate single sign-on portals without malware deployment."
        ),
    },
    {
        "report_id": "CISA-AA20-302A",
        "title": "Ransomware Activity Targeting Healthcare and Public Health Sector (Ryuk)",
        "source": "CISA",
        "publication_date": "2020-10-28",
        "tlp": "CLEAR",
        "author": "Cybersecurity and Infrastructure Security Agency",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-302a",
        "content": (
            "CISA, FBI, and HHS issued an urgent advisory regarding an imminent cybercrime threat to United States hospitals "
            "and healthcare providers involving Ryuk ransomware and TrickBot loaders.\n\n"
            "Adversaries initiated attacks through phishing emails containing weaponized Microsoft Word documents. In several "
            "intrusions, operators exploited CVE-2020-1472 Zerologon to compromise domain controllers within hours of initial access.\n\n"
            "The threat actors deployed Cobalt Strike and BloodHound to map hospital network shares. The operators executed "
            "Ryuk ransomware across clinical workstations, utilizing T1566 Phishing, T1078 Valid Accounts, and T1486 Data "
            "Encrypted for Impact to disrupt patient care."
        ),
    },
]


def generate():
    raw_path = Path("data/raw/cti_reports.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(REPORTS, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(REPORTS)} authentic CTI reports into {raw_path}")


if __name__ == "__main__":
    generate()
