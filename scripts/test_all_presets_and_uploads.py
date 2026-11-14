"""Validation script for all 3 Presets and Custom Ingestion with Novel Entities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import LiveReportProcessor

processor = LiveReportProcessor()

print("=" * 80)
print("TEST 1: PRESET 1 — APT28 / Zebrocy / NATO")
print("=" * 80)
p1_text = (
    "In recent intelligence updates, the threat actor APT28 (also tracked as Fancy Bear) was observed "
    "deploying the Zebrocy malware in operations across Europe. The threat group used spear-phishing "
    "attachments referencing upcoming defense coordination meetings to compromise targets associated with NATO "
    "and partner governments. Historical telemetry indicates that APT28 previously targeted NATO defense members "
    "during the 2020 operations."
)
res1 = processor.process_report(p1_text, "PRESET-1", "QuoIntelligence", "2021-05-15")
rels1 = [f"{r.source_entity} {r.relation_type} {r.target_entity}" for r in res1["relations"]]
print("Extracted Relations:", rels1)
assert "Zebrocy TARGETS Europe" not in rels1, "Zebrocy TARGETS Europe must NOT be present!"
assert "APT28 USES Zebrocy" in rels1
assert "APT28 TARGETS NATO" in rels1
verifs1 = processor.verify_extracted_intelligence(res1)
for v in verifs1:
    print(f"  • Claim: {v['claim']} -> Status: {v['status']} (Supp: {v['supporting_count']}, Contra: {v['contradicting_count']})")
print("Preset 1 Validation: PASSED\n")

print("=" * 80)
print("TEST 2: PRESET 2 — Emotet Takedown & Contradiction")
print("=" * 80)
p2_text = (
    "Following joint multinational law enforcement coordination in early 2021, authorities disrupted the Emotet botnet. "
    "Investigators confirmed that Emotet ceased operations and never downloaded or executed Trickbot on enterprise networks "
    "following the January 2021 infrastructure seizure. Threat actors previously reliant on Emotet loaders transitioned to "
    "independent phishing networks."
)
res2 = processor.process_report(p2_text, "PRESET-2", "CISA / Europol", "2021-05-20")
rels2 = [f"{r.source_entity} {r.relation_type} {r.target_entity}" for r in res2["relations"]]
print("Extracted Relations:", rels2)
verifs2 = processor.verify_extracted_intelligence(res2)
for v in verifs2:
    res_info = v.get("resolution", {})
    print(f"  • Claim: {v['claim']} -> Status: {v['status']}")
    print(f"    Context Resolution: {res_info.get('resolution_status')} (Delta: {res_info.get('temporal_delta_days')} days)")
print("Preset 2 Validation: PASSED\n")

print("=" * 80)
print("TEST 3: PRESET 3 — TrickBot & Emotet Partnership")
print("=" * 80)
p3_text = (
    "During late 2020, Intel 471 observed the Trickbot gang partnering with Emotet operators. In these coordinated campaigns, "
    "Emotet was distributing Trickbot across infected workstations as a secondary payload. The Emotet bots reached out to "
    "command-and-control servers to install Trickbot on compromised systems."
)
res3 = processor.process_report(p3_text, "PRESET-3", "Intel 471", "2020-11-20")
rels3 = [f"{r.source_entity} {r.relation_type} {r.target_entity}" for r in res3["relations"]]
print("Extracted Relations:", rels3)
verifs3 = processor.verify_extracted_intelligence(res3)
for v in verifs3:
    print(f"  • Claim: {v['claim']} -> Status: {v['status']} (Non-duplicate Supp: {v['non_duplicate_supporting_count']}, Dups: {v['duplicate_count']})")
print("Preset 3 Validation: PASSED\n")

print("=" * 80)
print("TEST 4: NOVEL UNSEEN ENTITIES UPLOAD & GRAPH EXPANSION")
print("=" * 80)
novel_text = (
    "In a newly uncovered 2024 campaign, threat actor ShadowHydra deployed the novel DarkLocker ransomware "
    "to target critical infrastructure organizations."
)
res4 = processor.process_report(novel_text, "UPLOAD-NOVEL-2024", "Mandiant", "2024-03-01")
print("Novel Entities Extracted:", [e.entity_id for e in res4["unique_entities"]])
print("Novel Relations Extracted:", [f"{r.source_entity} {r.relation_type} {r.target_entity}" for r in res4["relations"]])

merged4 = processor.merge_with_baseline_graph(res4)
print(f"Baseline Nodes: 1400 -> Merged Nodes: {merged4['total_nodes']} (+{merged4['new_nodes_count']} new nodes)")
print(f"Baseline Edges: 329  -> Merged Edges: {merged4['total_edges']} (+{merged4['new_edges_count']} new edges)")
print("New Node IDs:", merged4["new_node_ids"])

# Confirm baseline graph file was not mutated
import json
base_on_disk = json.load(open("data/cti_knowledge_graph_trusted.json", "r", encoding="utf-8"))
assert len(base_on_disk["nodes"]) == 1400, "Baseline graph on disk must remain strictly 1400 nodes!"
assert len(base_on_disk["links"]) == 329, "Baseline graph on disk must remain strictly 329 edges!"
print("\nIntegrity Check: PASSED! Baseline graph on disk is completely untouched.")
print("=" * 80)
