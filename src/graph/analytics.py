"""Cyber Knowledge Graph Analytics and Metrics.

Calculates graph centrality, threat actor profiling, vulnerability exposures,
and structural statistics from the Knowledge Graph.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import networkx as nx

logger = logging.getLogger(__name__)


class CyberKnowledgeGraphAnalytics:
    """Computes intelligence analytics on the Cyber Knowledge Graph."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def compute_summary(self) -> Dict[str, Any]:
        """Compute full statistical summary and intelligence metrics."""
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()

        # Category distribution
        node_categories: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            cat = data.get("category", "Unknown")
            node_categories[cat] = node_categories.get(cat, 0) + 1

        # Edge relation distribution
        edge_relations: Dict[str, int] = {}
        for _, _, key in self.graph.edges(keys=True):
            edge_relations[key] = edge_relations.get(key, 0) + 1

        # Centrality analysis (using simple directed graph view)
        di_graph = nx.DiGraph(self.graph)
        out_degrees = dict(di_graph.out_degree())
        in_degrees = dict(di_graph.in_degree())
        degree_centrality = nx.degree_centrality(di_graph)

        # Top Threat Actors by out-degree (activity)
        threat_actors = []
        for n, data in self.graph.nodes(data=True):
            if data.get("category") == "ThreatActor":
                threat_actors.append({
                    "name": data.get("name"),
                    "canonical_id": data.get("canonical_id"),
                    "out_degree": out_degrees.get(n, 0),
                    "in_degree": in_degrees.get(n, 0),
                    "mention_count": data.get("mention_count", 0),
                    "centrality": round(degree_centrality.get(n, 0.0), 4),
                    "reports": data.get("reports", []),
                    "first_seen": data.get("first_seen", "UNKNOWN"),
                    "last_seen": data.get("last_seen", "UNKNOWN"),
                })
        threat_actors.sort(key=lambda x: (x["out_degree"], x["mention_count"]), reverse=True)

        # Top Exploited CVEs by in-degree
        cves = []
        for n, data in self.graph.nodes(data=True):
            if data.get("category") == "CVE":
                cves.append({
                    "name": data.get("name"),
                    "exploited_by_count": in_degrees.get(n, 0),
                    "mention_count": data.get("mention_count", 0),
                    "reports": data.get("reports", []),
                })
        cves.sort(key=lambda x: (x["exploited_by_count"], x["mention_count"]), reverse=True)

        # Top Targeted Organizations & Locations
        targets = []
        for n, data in self.graph.nodes(data=True):
            if data.get("category") in ("Organization", "Location"):
                targets.append({
                    "name": data.get("name"),
                    "category": data.get("category"),
                    "targeted_by_count": in_degrees.get(n, 0),
                    "mention_count": data.get("mention_count", 0),
                    "reports": data.get("reports", []),
                })
        targets.sort(key=lambda x: (x["targeted_by_count"], x["mention_count"]), reverse=True)

        # Top Techniques
        techniques = []
        for n, data in self.graph.nodes(data=True):
            if data.get("category") == "Technique":
                techniques.append({
                    "name": data.get("name"),
                    "used_by_count": in_degrees.get(n, 0),
                    "mention_count": data.get("mention_count", 0),
                })
        techniques.sort(key=lambda x: (x["used_by_count"], x["mention_count"]), reverse=True)

        # Top Tools & Malware
        tools_malware = []
        for n, data in self.graph.nodes(data=True):
            if data.get("category") in ("Tool", "Malware"):
                tools_malware.append({
                    "name": data.get("name"),
                    "category": data.get("category"),
                    "used_by_count": in_degrees.get(n, 0),
                    "mention_count": data.get("mention_count", 0),
                })
        tools_malware.sort(key=lambda x: (x["used_by_count"], x["mention_count"]), reverse=True)

        summary = {
            "graph_metrics": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "node_categories": node_categories,
                "edge_relations": edge_relations,
            },
            "threat_actors_profiled": threat_actors,
            "top_exploited_cves": cves,
            "top_targets": targets,
            "top_techniques": techniques,
            "top_tools_malware": tools_malware,
        }
        return summary

    def export_summary(self, output_path: str = "outputs/reports/cti_kg_summary.json") -> Path:
        """Export analytics summary to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = self.compute_summary()
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported graph analytics summary to {out}")
        return out
