"""Cyber Knowledge Graph Exporter.

Exports the graph to:
- JSON node-link representation with top-level provenance fields
- GraphML format for network visualization tools (Gephi, Cytoscape)
- Neo4j Cypher script with schema constraints and MERGE operations
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import networkx as nx

logger = logging.getLogger(__name__)


class CyberKnowledgeGraphExporter:
    """Exports Cyber KG to multiple interoperable data formats."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def export_json(self, output_path: str = "data/knowledge_graph/cyber_kg.json") -> Path:
        """Export graph to standard JSON format with nodes and links preserving provenance."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            node_dict = {
                "id": node_id,
                "node_type": "ENTITY",
                "name": data.get("name", node_id),
                "category": data.get("category", "Entity"),
                "canonical_id": data.get("canonical_id", ""),
                "mention_count": data.get("mention_count", 1),
                # Top-level provenance
                "report_id": data.get("report_id", ""),
                "source": data.get("source", ""),
                "publication_date": data.get("publication_date", data.get("first_seen", "")),
                "sentence_id": data.get("sentence_id", ""),
                "first_seen": data.get("first_seen", "UNKNOWN"),
                "last_seen": data.get("last_seen", "UNKNOWN"),
                "reports": list(data.get("reports", [])),
                "sources": list(data.get("sources", [])),
                "dates": list(data.get("dates", [])),
                "surface_forms": list(data.get("surface_forms", [])),
            }
            nodes.append(node_dict)

        links = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            evidence_list = data.get("evidence", [])
            primary_ev = evidence_list[0] if evidence_list else {}

            link_dict = {
                "source": u,
                "target": v,
                "key": key,
                "relation_type": data.get("relation_type", key),
                "confidence": data.get("confidence", 1.0),
                "confidence_level": data.get("confidence_level", "HIGH"),
                "polarity": data.get("polarity", "positive"),
                "observation_time": data.get("observation_time", primary_ev.get("publication_date", "")),
                "scope": data.get("scope", "sentence_level"),
                "evidence_count": data.get("evidence_count", 1),
                # Top-level provenance fields directly on the edge
                "report_id": data.get("report_id", primary_ev.get("report_id", "")),
                "source_publisher": data.get("source", primary_ev.get("source", "")),
                "publication_date": data.get("publication_date", primary_ev.get("publication_date", "")),
                "sentence_id": data.get("sentence_id", primary_ev.get("sentence_id", "")),
                "evidence_text": data.get("evidence_text", primary_ev.get("evidence_text", "")),
                "trigger": data.get("trigger", primary_ev.get("trigger", "")),
                "reports": list(data.get("reports", [])),
                "sources": list(data.get("sources", [])),
                "dates": list(data.get("dates", [])),
                "evidence": evidence_list,
            }
            links.append(link_dict)

        payload = {
            "graph_metadata": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
            },
            "nodes": nodes,
            "links": links,
        }

        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported JSON graph with {len(nodes)} nodes and {len(links)} links to {out}")
        return out

    def export_graphml(self, output_path: str = "data/knowledge_graph/cyber_kg.graphml") -> Path:
        """Export graph to GraphML format."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        g_copy = nx.MultiDiGraph()
        for node_id, data in self.graph.nodes(data=True):
            clean_attrs = {}
            for k, val in data.items():
                if isinstance(val, (list, set)):
                    clean_attrs[k] = ", ".join(str(x) for x in val)
                elif isinstance(val, dict):
                    clean_attrs[k] = json.dumps(val)
                else:
                    clean_attrs[k] = str(val)
            g_copy.add_node(node_id, **clean_attrs)

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            clean_edge = {}
            for attr_name, attr_val in data.items():
                if attr_name == "evidence":
                    clean_edge[attr_name] = json.dumps(attr_val)
                elif isinstance(attr_val, (list, set)):
                    clean_edge[attr_name] = ", ".join(str(x) for x in attr_val)
                else:
                    clean_edge[attr_name] = str(attr_val)
            g_copy.add_edge(u, v, key=key, **clean_edge)

        nx.write_graphml(g_copy, out)
        logger.info(f"Exported GraphML graph to {out}")
        return out

    def export_cypher(self, output_path: str = "data/knowledge_graph/import_kg.cypher") -> Path:
        """Generate Neo4j Cypher script to import the entire knowledge graph."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "// ====================================================",
            "// Cyber Knowledge Graph Import Script for Neo4j",
            "// Generated by CyberRAG Pipeline",
            "// ====================================================\n",
            "// 1. Create Schema Constraints",
            "CREATE CONSTRAINT threat_actor_name IF NOT EXISTS FOR (n:ThreatActor) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT malware_name IF NOT EXISTS FOR (n:Malware) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT tool_name IF NOT EXISTS FOR (n:Tool) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT cve_name IF NOT EXISTS FOR (n:CVE) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT technique_name IF NOT EXISTS FOR (n:Technique) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT organization_name IF NOT EXISTS FOR (n:Organization) REQUIRE n.name IS UNIQUE;",
            "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (n:Location) REQUIRE n.name IS UNIQUE;\n",
            "// 2. Merge Nodes",
        ]

        # Add Nodes
        for node_id, data in self.graph.nodes(data=True):
            cat = data.get("category", "Entity")
            name = data.get("name", "").replace("'", "\\'")
            cid = data.get("canonical_id", "").replace("'", "\\'")
            mention_count = data.get("mention_count", 1)
            first_seen = data.get("first_seen", "UNKNOWN")
            last_seen = data.get("last_seen", "UNKNOWN")
            report_id = data.get("report_id", "").replace("'", "\\'")
            source = data.get("source", "").replace("'", "\\'")
            reports = json.dumps(data.get("reports", []))

            stmt = (
                f"MERGE (n:{cat} {{name: '{name}'}}) "
                f"ON CREATE SET n.canonical_id = '{cid}', n.mention_count = {mention_count}, "
                f"n.first_seen = '{first_seen}', n.last_seen = '{last_seen}', "
                f"n.report_id = '{report_id}', n.source = '{source}', "
                f"n.publication_date = '{first_seen}', n.reports = {reports} "
                f"ON MATCH SET n.mention_count = {mention_count};"
            )
            lines.append(stmt)

        lines.append("\n// 3. Merge Relationships")
        # Add Edges
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            u_node = self.graph.nodes[u]
            v_node = self.graph.nodes[v]

            u_cat = u_node.get("category", "Entity")
            u_name = u_node.get("name", "").replace("'", "\\'")

            v_cat = v_node.get("category", "Entity")
            v_name = v_node.get("name", "").replace("'", "\\'")

            rel_type = data.get("relation_type", key)
            conf = data.get("confidence", 0.9)
            ev_count = data.get("evidence_count", 1)

            rep_id = data.get("report_id", "").replace("'", "\\'")
            src = data.get("source", "").replace("'", "\\'")
            pub_date = data.get("publication_date", "").replace("'", "\\'")
            sent_id = data.get("sentence_id", "").replace("'", "\\'")

            stmt = (
                f"MATCH (a:{u_cat} {{name: '{u_name}'}}), (b:{v_cat} {{name: '{v_name}'}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                f"ON CREATE SET r.confidence = {conf:.2f}, r.evidence_count = {ev_count}, "
                f"r.polarity = 'positive', r.observation_time = '{pub_date}', "
                f"r.report_id = '{rep_id}', r.source = '{src}', "
                f"r.publication_date = '{pub_date}', r.sentence_id = '{sent_id}';"
            )
            lines.append(stmt)

        content = "\n".join(lines) + "\n"
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Exported Neo4j Cypher script to {out}")
        return out
