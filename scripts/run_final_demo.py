"""Final Demonstration of the CyberRAG End-to-End System.

Showcases three core scenarios:
1. Multi-Source Corroborated Relationship: 'How was TrickBot related to Emotet?'
2. Threat Actor Targeting Assertion: 'Did APT28 target NATO?'
3. Opposition & Context-Aware Contradiction Resolution: Contradictory proposition vs. real CTI evidence.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_rag_pipeline import GraphRAGPipeline


def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_claim_summary(claim_data: dict):
    print(f"  Target Proposition : {claim_data.get('claim')}")
    print(f"  NLI Aggregated Status: {claim_data.get('status')}")
    print(f"  Total Evidence Evaluated: {claim_data.get('total_evidence')} passages")
    print(f"  Supporting Count   : {claim_data.get('supporting_count')} (Non-duplicate: {claim_data.get('non_duplicate_supporting_count')})")
    print(f"  Contradicting Count: {claim_data.get('contradicting_count')}")
    print(f"  Neutral Count      : {claim_data.get('neutral_count')}")
    print(f"  Duplicate Records  : {claim_data.get('duplicate_count')} (flagged to prevent false corroboration)")
    print(f"  Unique Sources     : {', '.join(claim_data.get('unique_supporting_sources', []))}")

    # Context resolution
    res = claim_data.get("resolution", {})
    if res:
        print(f"\n  [CTI Context Resolution]")
        print(f"  Resolution Status  : {res.get('resolution_status')}")
        print(f"  Explanation        : {res.get('explanation')}")
        if res.get("temporal_delta_days") is not None:
            print(f"  Temporal Gap       : {res.get('temporal_delta_days')} days ({res.get('temporal_relationship')})")
        print(f"  Recommended Action : {res.get('recommended_action')}")

    print("\n  [Evidence Provenance Samples]")
    for idx, ev in enumerate(claim_data.get("evidence_results", [])[:4], 1):
        dup_tag = f" [DUPLICATE of {ev.get('duplicate_of')}]" if ev.get("is_duplicate") else ""
        print(f"    {idx}. [{ev.get('evidence_type').upper()}]{dup_tag} {ev.get('source')} ({ev.get('publication_date')})")
        print(f"       Label: {ev.get('label').upper()} (p_ent={ev.get('p_entailment')}, p_contra={ev.get('p_contradiction')}, p_neu={ev.get('p_neutral')})")
        text_preview = ev.get('evidence', '').strip().replace('\n', ' ')
        if len(text_preview) > 130:
            text_preview = text_preview[:127] + "..."
        print(f"       Snippet: \"{text_preview}\"")


def run_demo():
    print_separator("CYBERRAG PIPELINE INITIALIZATION")
    pipeline = GraphRAGPipeline()
    print("All subsystems online: Preprocessing -> NER -> Knowledge Graph -> Hybrid RAG -> NLI Verifier -> Context Resolver.")

    # Demo 1
    print_separator("DEMO 1: CORROBORATED MALWARE DELIVERY RELATIONSHIP")
    q1 = "How was TrickBot related to Emotet?"
    print(f"Query: {q1}\n")
    res1 = pipeline.ask(q1, verify=True)
    print(f"[Generated Answer]:\n{res1['answer']}\n")
    print("[Verification Summary]:")
    for claim in res1["verification"]["claims"]:
        print_claim_summary(claim)

    # Demo 2
    print_separator("DEMO 2: THREAT ACTOR TARGETING ASSERTION")
    q2 = "Did APT28 target NATO?"
    print(f"Query: {q2}\n")
    res2 = pipeline.ask(q2, verify=True)
    print(f"[Generated Answer]:\n{res2['answer']}\n")
    print("[Verification Summary]:")
    for claim in res2["verification"]["claims"]:
        print_claim_summary(claim)

    # Demo 3
    print_separator("DEMO 3: CONTRADICTION & CONTEXT-AWARE RESOLUTION")
    q3 = "How was TrickBot related to Emotet?"
    hypo3 = ["Emotet never downloaded or executed Trickbot."]
    print(f"Query: {q3}")
    print(f"Contradictory Hypothesis Test: {hypo3[0]}\n")
    res3 = pipeline.ask(q3, verify=True, verification_claims=hypo3)
    print(f"[Generated Answer]:\n{res3['answer']}\n")
    print("[Verification Summary]:")
    for claim in res3["verification"]["claims"]:
        print_claim_summary(claim)

    print_separator("DEMO EXECUTION COMPLETE")


if __name__ == "__main__":
    run_demo()
