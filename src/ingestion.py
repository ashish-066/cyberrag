"""Live CTI Report Ingestion and Graph Update Module.

Handles:
1. Ingestion of arbitrary PDF / TXT threat reports.
2. Text cleaning and provenance-preserving sentence segmentation.
3. Cybersecurity NER and Relation Extraction.
4. In-memory Knowledge Graph merging (without mutating baseline files).
5. PyVis/Vis.js interactive network generation highlighting new intelligence.
6. Evidence verification and context resolution against existing 104-report corpus.
"""
from __future__ import annotations
import copy
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF
from pyvis.network import Network

from src.preprocessor.cleaner import CTICleaner
from src.preprocessor.segmenter import CTISegmenter, PreprocessedSentence, PreprocessedReport
from src.nlp.ner_engine import CyberNEREngine, ExtractedEntity
from src.nlp.relation_extractor import CyberRelationExtractor, ExtractedRelation
from src.graph_rag_pipeline import GraphRAGPipeline

logger = logging.getLogger(__name__)


CATEGORY_COLORS = {
    "ThreatActor": "#e63946",       # Red
    "Malware": "#7209b7",           # Purple
    "Tool": "#f77f00",              # Orange
    "CVE": "#fcbf49",               # Yellow
    "Technique": "#4361ee",         # Blue
    "Organization": "#2a9d8f",      # Green
    "Location": "#457b9d",          # Teal
}
NEW_NODE_COLOR = "#00f5d4"          # Neon Cyan for newly ingested nodes
NEW_EDGE_COLOR = "#00f5d4"          # Neon Cyan for newly extracted edges


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """Extract clean raw text from an uploaded PDF or TXT file."""
    fn_lower = filename.lower()
    if fn_lower.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        return "\n\n".join(pages)
    else:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="replace")


class LiveReportProcessor:
    """Processes uploaded CTI reports through NER, RE, Graph update, and verification."""

    def __init__(
        self,
        graph_path: str = "data/cti_knowledge_graph_trusted.json",
        pipeline: Optional[GraphRAGPipeline] = None,
    ):
        self.cleaner = CTICleaner()
        self.segmenter = CTISegmenter(cleaner=self.cleaner)
        self.ner_engine = CyberNEREngine(enable_ml=True)
        self.relation_extractor = CyberRelationExtractor()
        self.pipeline = pipeline or GraphRAGPipeline()

        # Load baseline knowledge graph in-memory (immutable reference)
        with open(graph_path, "r", encoding="utf-8") as f:
            self.baseline_graph: Dict[str, Any] = json.load(f)

    def process_report(
        self,
        raw_text: str,
        report_id: str,
        source: str,
        pub_date: str,
        title: str = "Uploaded CTI Report",
    ) -> Dict[str, Any]:
        """Run complete extraction on uploaded report text."""
        # Clean text
        clean_text = self.cleaner.clean(raw_text)

        # Segment into paragraphs and sentences
        paragraphs = clean_text.split("\n\n")
        sentences: List[PreprocessedSentence] = []
        global_idx = 0

        for p_idx, p_text in enumerate(paragraphs):
            p_text_clean = p_text.strip()
            if not p_text_clean:
                continue
            p_id = f"{report_id}::p{p_idx}"
            segmented_sents = self.segmenter.segment_sentences(p_text_clean)
            for s_idx, (s_text, s_start, s_end) in enumerate(segmented_sents):
                s_id = f"{p_id}::s{s_idx}"
                sentences.append(
                    PreprocessedSentence(
                        sentence_id=s_id,
                        paragraph_id=p_id,
                        report_id=report_id,
                        source=source,
                        publication_date=pub_date,
                        paragraph_index=p_idx,
                        sentence_index=s_idx,
                        global_sentence_index=global_idx,
                        text=s_text,
                        start_char=s_start,
                        end_char=s_end,
                    )
                )
                global_idx += 1

        # Extract entities per sentence
        all_entities: List[ExtractedEntity] = []
        entity_map_by_sentence: Dict[str, List[ExtractedEntity]] = {}

        for sent in sentences:
            sent_entities = self.ner_engine.extract_from_sentence(sent)
            all_entities.extend(sent_entities)
            entity_map_by_sentence[sent.sentence_id] = sent_entities

        # Extract relations per sentence
        all_relations: List[ExtractedRelation] = []
        for sent in sentences:
            sent_entities = entity_map_by_sentence.get(sent.sentence_id, [])
            if len(sent_entities) >= 2:
                sent_relations = self.relation_extractor.extract_from_sentence(sent, sent_entities)
                all_relations.extend(sent_relations)

        # Unique entities and category tally
        unique_entities_dict: Dict[str, ExtractedEntity] = {}
        category_counts: Dict[str, int] = {}
        for ent in all_entities:
            unique_entities_dict[ent.entity_id] = ent
            category_counts[ent.category] = category_counts.get(ent.category, 0) + 1

        unique_entities = list(unique_entities_dict.values())
        positive_relations = [r for r in all_relations if getattr(r, "polarity", "positive") == "positive"]
        negative_assertions = [r for r in all_relations if getattr(r, "polarity", "positive") == "negative"]

        return {
            "report_id": report_id,
            "title": title,
            "source": source,
            "publication_date": pub_date,
            "total_characters": len(clean_text),
            "total_sentences": len(sentences),
            "total_entities_extracted": len(all_entities),
            "unique_entities_count": len(unique_entities),
            "category_counts": category_counts,
            "unique_entities": unique_entities,
            "relations": all_relations,
            "positive_relations": positive_relations,
            "negative_assertions": negative_assertions,
            "sentences": sentences,
        }

    def merge_with_baseline_graph(
        self,
        extraction_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create an updated in-memory Knowledge Graph incorporating the new intelligence."""
        merged_graph = copy.deepcopy(self.baseline_graph)
        existing_nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in merged_graph.get("nodes", [])}
        existing_edges: Set[Tuple[str, str, str]] = {
            (e["source"], e["target"], e.get("relation_type", e.get("key", "")))
            for e in merged_graph.get("links", [])
        }

        new_node_ids: Set[str] = set()
        new_edge_keys: Set[Tuple[str, str, str]] = set()

        # Merge extracted entities
        for ent in extraction_result.get("unique_entities", []):
            node_id = ent.entity_id
            if node_id in existing_nodes:
                node = existing_nodes[node_id]
                node["mention_count"] = node.get("mention_count", 1) + 1
                if extraction_result["report_id"] not in node.get("reports", []):
                    node.setdefault("reports", []).append(extraction_result["report_id"])
                if extraction_result["source"] not in node.get("sources", []):
                    node.setdefault("sources", []).append(extraction_result["source"])
                node["is_new"] = False
                node["updated_by_upload"] = True
            else:
                new_node = {
                    "id": node_id,
                    "node_type": "ENTITY",
                    "name": ent.canonical_name,
                    "category": ent.category,
                    "canonical_id": ent.canonical_id,
                    "mention_count": 1,
                    "report_id": extraction_result["report_id"],
                    "source": extraction_result["source"],
                    "publication_date": extraction_result["publication_date"],
                    "reports": [extraction_result["report_id"]],
                    "sources": [extraction_result["source"]],
                    "dates": [extraction_result["publication_date"]],
                    "surface_forms": [ent.text],
                    "is_new": True,
                }
                merged_graph["nodes"].append(new_node)
                existing_nodes[node_id] = new_node
                new_node_ids.add(node_id)

        # Merge extracted relations
        for rel in extraction_result.get("relations", []):
            # SAFEGUARD 1: Never turn a negative assertion into a positive KG edge!
            if getattr(rel, "polarity", "positive") == "negative":
                continue

            src_id = f"{rel.source_category}::{rel.source_entity}"
            tgt_id = f"{rel.target_category}::{rel.target_entity}"
            edge_key = (src_id, tgt_id, rel.relation_type)

            if edge_key not in existing_edges:
                new_edge = {
                    "source": src_id,
                    "target": tgt_id,
                    "key": rel.relation_type,
                    "relation_type": rel.relation_type,
                    "confidence": rel.confidence,
                    "confidence_level": "HIGH" if rel.confidence >= 0.8 else "MEDIUM",
                    "polarity": rel.polarity,
                    "evidence_text": rel.evidence_text,
                    "trigger": rel.trigger,
                    "report_id": rel.report_id,
                    "source_publisher": rel.source,
                    "publication_date": rel.publication_date,
                    "sentence_id": rel.sentence_id,
                    "reports": [rel.report_id],
                    "sources": [rel.source],
                    "dates": [rel.publication_date],
                    "is_new": True,
                }
                merged_graph["links"].append(new_edge)
                existing_edges.add(edge_key)
                new_edge_keys.add(edge_key)

        return {
            "graph": merged_graph,
            "new_node_ids": list(new_node_ids),
            "new_edge_keys": [f"{s} -[{r}]-> {t}" for s, t, r in new_edge_keys],
            "total_nodes": len(merged_graph["nodes"]),
            "total_edges": len(merged_graph["links"]),
            "new_nodes_count": len(new_node_ids),
            "new_edges_count": len(new_edge_keys),
        }

    def verify_extracted_intelligence(
        self,
        extraction_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Verify each extracted relation proposition against existing 104-report corpus."""
        relations = extraction_result.get("relations", [])
        if not relations:
            return []

        # Deduplicate candidate claims strictly (including polarity)
        claims_to_verify: Dict[str, Tuple[str, ExtractedRelation]] = {}
        for rel in relations:
            claim_text = f"{rel.source_entity.strip()} {rel.relation_type.strip()} {rel.target_entity.strip()}."
            pol = getattr(rel, "polarity", "positive")
            key = f"{claim_text.strip().lower()}::{pol}"
            if key not in claims_to_verify:
                claims_to_verify[key] = (claim_text, rel)

        verification_reports: List[Dict[str, Any]] = []

        for key, (claim_text, rel) in claims_to_verify.items():
            # Query pipeline for corroboration & conflict in existing corpus
            rag_output = self.pipeline.ask(
                query=claim_text,
                verify=True,
                verification_claims=[claim_text],
            )
            verif = rag_output.get("verification", {})
            claims_data = verif.get("claims", [])
            claim_details = claims_data[0] if claims_data else {}

            evidence_results = list(claim_details.get("evidence_results", []))
            status = claim_details.get("status", verif.get("status", "UNVERIFIED"))
            supporting_count = claim_details.get("supporting_count", 0)
            non_duplicate_supporting_count = claim_details.get("non_duplicate_supporting_count", 0)
            contradicting_count = claim_details.get("contradicting_count", 0)
            neutral_count = claim_details.get("neutral_count", 0)
            duplicate_count = claim_details.get("duplicate_count", 0)
            unique_supporting_sources = list(claim_details.get("unique_supporting_sources", []))
            resolution = claim_details.get("resolution", {})

            # When the extracted intelligence is a negative assertion (e.g. post-takedown operational report):
            if getattr(rel, "polarity", "positive") == "negative":
                if not self.pipeline.verifier:
                    raise RuntimeError(
                        "Cannot verify negative intelligence upload: CTIEvidenceVerifier (DeBERTa-v3) is not loaded."
                    )
                nli_pred = self.pipeline.verifier.classify_pair(evidence=rel.evidence_text, claim=claim_text)
                pred_label = nli_pred.label
                p_ent = round(nli_pred.p_entailment, 4)
                p_contra = round(nli_pred.p_contradiction, 4)
                p_neut = round(nli_pred.p_neutral, 4)

                uploaded_ev_record = {
                    "label": pred_label,
                    "p_entailment": p_ent,
                    "p_contradiction": p_contra,
                    "p_neutral": p_neut,
                    "evidence": rel.evidence_text,
                    "claim": claim_text,
                    "evidence_id": f"UPLOAD::{rel.relation_id}",
                    "evidence_type": "uploaded_report",
                    "assertion_id": None,
                    "report_id": rel.report_id,
                    "segment_id": rel.sentence_id,
                    "source": rel.source,
                    "publication_date": rel.publication_date,
                    "confidence": rel.confidence,
                    "trust_level": "REPORTED",
                    "is_duplicate": False,
                    "duplicate_of": None,
                }

                # Combine historical corpus evidence with the newly ingested negative evidence
                fused_evidence = evidence_results + [uploaded_ev_record]

                # Run ContextResolver to evaluate temporal divergence across fused evidence
                from src.verification.context_resolver import ContextResolver
                res_obj = ContextResolver.resolve(claim=claim_text, evidence_results=fused_evidence)
                resolution = res_obj.to_dict()

                # Use pipeline's ClaimAggregator to produce consistent status verdict
                agg = self.pipeline.claim_aggregator.aggregate(claim=claim_text, results=fused_evidence)
                agg_status = agg.status.value if hasattr(agg.status, "value") else str(agg.status)
                status = "CONFLICTING_EVIDENCE" if res_obj.is_conflict else agg_status
                supporting_count = agg.supporting_count
                non_duplicate_supporting_count = len([e for e in fused_evidence if e.get("label") == "entailment" and not e.get("is_duplicate")])
                contradicting_count = agg.contradicting_count
                neutral_count = agg.neutral_count
                unique_supporting_sources = sorted(list({e.get("source") for e in fused_evidence if e.get("label") == "entailment" and e.get("source")}))
                evidence_results = fused_evidence

            verification_reports.append({
                "claim": claim_text,
                "extracted_from_sentence": rel.evidence_text,
                "trigger": rel.trigger,
                "confidence": rel.confidence,
                "polarity": getattr(rel, "polarity", "positive"),
                "status": status,
                "total_evidence": len(evidence_results),
                "supporting_count": supporting_count,
                "non_duplicate_supporting_count": non_duplicate_supporting_count,
                "contradicting_count": contradicting_count,
                "neutral_count": neutral_count,
                "duplicate_count": duplicate_count,
                "unique_supporting_sources": unique_supporting_sources,
                "resolution": resolution,
                "evidence_results": evidence_results,
            })

        return verification_reports

    def generate_subgraph_html(
        self,
        merged_graph: Dict[str, Any],
        focus_entity_ids: List[str],
        max_hops: int = 1,
        height: str = "750px",
    ) -> str:
        """Generate interactive Vis.js HTML centered on uploaded report entities and their neighborhood."""
        nodes_dict = {n["id"]: n for n in merged_graph.get("nodes", [])}
        links = merged_graph.get("links", [])

        # Build adjacency for ego network extraction
        sub_nodes: Set[str] = set(focus_entity_ids)

        for _ in range(max_hops):
            current = set(sub_nodes)
            for link in links:
                s, t = link["source"], link["target"]
                if s in current or t in current:
                    sub_nodes.add(s)
                    sub_nodes.add(t)

        # Fallback if no focus or isolated
        if len(sub_nodes) < 3 and len(nodes_dict) > 0:
            sub_nodes.update(list(nodes_dict.keys())[:25])

        # Limit to 50 nodes for smooth browser rendering
        if len(sub_nodes) > 60:
            sorted_nodes = sorted(
                sub_nodes,
                key=lambda nid: (nid in focus_entity_ids, nodes_dict.get(nid, {}).get("mention_count", 0)),
                reverse=True
            )
            sub_nodes = set(sorted_nodes[:60])

        net = Network(height=height, width="100%", bgcolor="#0d1117", font_color="#c9d1d9", directed=True)
        net.barnes_hut(gravity=-6000, central_gravity=0.2, spring_length=220, spring_strength=0.03, damping=0.1, overlap=0.6)

        for nid in sub_nodes:
            node = nodes_dict.get(nid)
            if not node:
                continue
            cat = node.get("category", "Unknown")
            is_new = node.get("is_new", False)

            color = NEW_NODE_COLOR if is_new else CATEGORY_COLORS.get(cat, "#8b949e")
            border_color = "#ffffff" if is_new else "#30363d"
            size = 28 if is_new else 18
            shape = "diamond" if is_new else "dot"

            title_html = (
                f"<b>{node.get('name')}</b><br>"
                f"Category: {cat}<br>"
                f"Status: {'NEW UPLOAD' if is_new else 'BASELINE'}<br>"
                f"Sources: {', '.join(node.get('sources', []))}<br>"
                f"Mentions: {node.get('mention_count', 1)}"
            )

            net.add_node(
                nid,
                label=f"{'✦ ' if is_new else ''}{node.get('name', nid)}",
                title=title_html,
                color={"background": color, "border": border_color, "highlight": {"background": "#ffffff", "border": color}},
                size=size,
                shape=shape,
                font={"size": 13, "color": "#f0f6fc", "face": "Segoe UI, Arial"},
            )

        for link in links:
            s, t = link["source"], link["target"]
            if s in sub_nodes and t in sub_nodes:
                is_new = link.get("is_new", False)
                edge_color = NEW_EDGE_COLOR if is_new else "#58a6ff"
                rel_type = link.get("relation_type", link.get("key", ""))
                net.add_edge(
                    s,
                    t,
                    label=rel_type,
                    color=edge_color,
                    arrows="to",
                    width=3 if is_new else 1,
                    smooth={"type": "curvedCW", "roundness": 0.2},
                    font={"size": 10, "color": "#f1f5f9", "background": "#0f172a", "strokeWidth": 0, "align": "middle"},
                    title=f"Trigger: {link.get('trigger', '')}<br>Evidence: {link.get('evidence_text', '')[:120]}...",
                )

        return net.generate_html()
