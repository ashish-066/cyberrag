import json
import pprint
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_rag_pipeline import GraphRAGPipeline

print("=== Initializing GraphRAGPipeline ===")
rag = GraphRAGPipeline()

print("\n=== Running rag.ask('How was TrickBot related to Emotet?', verify=True) ===")
result = rag.ask(
    "How was TrickBot related to Emotet?",
    verify=True
)

print("\n=== Verification Output ===")
pprint.pprint(result["verification"])

print("\n=== Verification JSON Summary ===")
# Pretty print summary without huge nested dumps
summary = {
    "status": result["verification"]["status"],
    "total_claims_evaluated": result["verification"]["total_claims_evaluated"],
    "claims": []
}
for c in result["verification"]["claims"]:
    c_summary = {
        "claim": c["claim"],
        "status": c["status"],
        "total_evidence": c["total_evidence"],
        "supporting_count": c["supporting_count"],
        "contradicting_count": c["contradicting_count"],
        "neutral_count": c["neutral_count"],
        "duplicate_count": c["duplicate_count"],
        "non_duplicate_supporting_count": c["non_duplicate_supporting_count"],
        "unique_supporting_sources": c["unique_supporting_sources"],
        "evidence_samples": [
            {
                "evidence_id": ev["evidence_id"],
                "evidence_type": ev["evidence_type"],
                "label": ev["label"],
                "p_entailment": ev["p_entailment"],
                "p_contradiction": ev["p_contradiction"],
                "p_neutral": ev["p_neutral"],
                "source": ev["source"],
                "publication_date": ev["publication_date"],
                "is_duplicate": ev.get("is_duplicate"),
                "duplicate_of": ev.get("duplicate_of"),
                "evidence_preview": (ev["evidence"][:120] + "...") if len(ev["evidence"]) > 120 else ev["evidence"],
            }
            for ev in c["evidence_results"]
        ]
    }
    summary["claims"].append(c_summary)

print(json.dumps(summary, indent=2))
