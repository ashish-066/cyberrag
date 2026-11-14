import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import LiveReportProcessor

processor = LiveReportProcessor()
text = (
    "In recent intelligence updates, the threat actor APT28 (also tracked as Fancy Bear) was observed "
    "deploying the Zebrocy malware in operations across Europe. The threat group used spear-phishing "
    "attachments referencing upcoming defense coordination meetings to compromise targets associated with NATO "
    "and partner governments. Historical telemetry indicates that APT28 previously targeted NATO defense members "
    "during the 2020 operations."
)

res = processor.process_report(text, "PRESET-1", "QuoIntelligence", "2021-05-15")
print("\n=== EXTRACTED RELATIONS ===")
for r in res["relations"]:
    print(f"  - {r.source_entity} {r.relation_type} {r.target_entity} (trigger: '{r.trigger}')")

print(f"\nTotal relations: {len(res['relations'])}")
assert not any(r.target_entity == "Europe" for r in res["relations"]), "Europe should NOT be extracted as TARGETS!"
assert any(r.source_entity == "APT28" and r.target_entity == "Zebrocy" and r.relation_type == "USES" for r in res["relations"]), "APT28 USES Zebrocy must be extracted!"
assert any(r.source_entity == "APT28" and r.target_entity == "NATO" and r.relation_type == "TARGETS" for r in res["relations"]), "APT28 TARGETS NATO must be extracted!"
print("\nAssertion Check: PASSED! Zero unsupported spatial relationships extracted.")
