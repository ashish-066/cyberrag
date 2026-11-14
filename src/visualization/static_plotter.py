"""Static Graph Plotter using Matplotlib and NetworkX.

Renders publication-ready static network diagrams for reports and presentations.
"""

import logging
from pathlib import Path
from typing import Dict, List
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

HEX_COLORS = {
    "ThreatActor": "#ef4444",
    "Malware": "#a855f7",
    "Tool": "#06b6d4",
    "CVE": "#f97316",
    "Technique": "#eab308",
    "Organization": "#10b981",
    "Location": "#3b82f6",
}

RELATION_EDGE_COLORS = {
    "USES": "#06b6d4",
    "TARGETS": "#10b981",
    "EXPLOITS": "#f97316",
    "USES_TECHNIQUE": "#eab308",
}


class StaticGraphPlotter:
    """Generates static network diagrams for CTI analysis."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def plot_full_graph(self, output_path: str = "outputs/visualizations/full_knowledge_graph.png") -> Path:
        """Render high-resolution full knowledge graph overview."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(20, 16), facecolor="#0b0f19")
        ax = plt.gca()
        ax.set_facecolor("#0b0f19")

        # Convert MultiDiGraph to simple DiGraph for spring layout
        di_g = nx.DiGraph(self.graph)
        pos = nx.spring_layout(di_g, k=0.45, iterations=80, seed=42)

        # Node colors & sizes
        node_colors = []
        node_sizes = []
        labels = {}
        for n, data in self.graph.nodes(data=True):
            cat = data.get("category", "Entity")
            deg = di_g.degree(n)
            node_colors.append(HEX_COLORS.get(cat, "#94a3b8"))
            base_size = 400 + (deg * 120)
            if cat == "ThreatActor":
                base_size += 300
            node_sizes.append(base_size)
            labels[n] = data.get("name", n)

        # Draw Nodes
        nx.draw_networkx_nodes(
            di_g,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.92,
            edgecolors="#1e293b",
            linewidths=1.5,
            ax=ax,
        )

        # Draw Edges with relation colors
        edge_colors = []
        for u, v in di_g.edges():
            # Get relation type from original multigraph
            edge_data = self.graph.get_edge_data(u, v, default={})
            rel_type = "USES"
            if edge_data:
                first_key = list(edge_data.keys())[0]
                rel_type = edge_data[first_key].get("relation_type", "USES")
            edge_colors.append(RELATION_EDGE_COLORS.get(rel_type, "#64748b"))

        nx.draw_networkx_edges(
            di_g,
            pos,
            edge_color=edge_colors,
            alpha=0.55,
            arrows=True,
            arrowsize=14,
            width=1.2,
            connectionstyle="arc3,rad=0.08",
            ax=ax,
        )

        # Draw Labels
        nx.draw_networkx_labels(
            di_g,
            pos,
            labels=labels,
            font_size=9,
            font_color="#f8fafc",
            font_family="sans-serif",
            font_weight="bold",
            ax=ax,
        )

        # Custom Legend
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=cat, markerfacecolor=col, markersize=12)
            for cat, col in HEX_COLORS.items()
        ]
        edge_legend_handles = [
            plt.Line2D([0], [0], color=col, lw=2.5, label=f"Edge: {rel}")
            for rel, col in RELATION_EDGE_COLORS.items()
        ]
        legend = plt.legend(
            handles=legend_handles + edge_legend_handles,
            loc="upper left",
            frameon=True,
            facecolor="#0f172a",
            edgecolor="#1e293b",
            fontsize=10,
            labelcolor="#e2e8f0",
        )
        legend.get_frame().set_alpha(0.85)

        plt.title("Cyber Knowledge Graph — Multi-Relational CTI Network", color="#38bdf8", fontsize=18, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(out, dpi=250, facecolor="#0b0f19")
        plt.close()

        logger.info(f"Saved full knowledge graph plot to {out}")
        return out

    def plot_threat_actor_subgraph(self, output_path: str = "outputs/visualizations/threat_actor_network.png") -> Path:
        """Render subgraph focused on Threat Actors and their 1-hop connections."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        di_g = nx.DiGraph(self.graph)
        actor_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("category") == "ThreatActor"]

        subgraph_nodes = set(actor_nodes)
        for act in actor_nodes:
            subgraph_nodes.update(di_g.successors(act))
            subgraph_nodes.update(di_g.predecessors(act))

        sub_g = di_g.subgraph(subgraph_nodes)

        plt.figure(figsize=(18, 14), facecolor="#0b0f19")
        ax = plt.gca()
        ax.set_facecolor("#0b0f19")

        pos = nx.spring_layout(sub_g, k=0.5, iterations=70, seed=123)

        node_colors = [HEX_COLORS.get(self.graph.nodes[n].get("category", "Entity"), "#94a3b8") for n in sub_g.nodes()]
        node_sizes = [700 if self.graph.nodes[n].get("category") == "ThreatActor" else 350 for n in sub_g.nodes()]
        labels = {n: self.graph.nodes[n].get("name", n) for n in sub_g.nodes()}

        nx.draw_networkx_nodes(sub_g, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors="#1e293b", ax=ax)
        nx.draw_networkx_edges(sub_g, pos, edge_color="#38bdf8", alpha=0.6, arrows=True, arrowsize=14, width=1.4, ax=ax)
        nx.draw_networkx_labels(sub_g, pos, labels=labels, font_size=8.5, font_color="#f8fafc", font_weight="bold", ax=ax)

        plt.title("Threat Actor Campaigns: Operations, Capabilities & Targets", color="#38bdf8", fontsize=16, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(out, dpi=250, facecolor="#0b0f19")
        plt.close()

        logger.info(f"Saved threat actor network plot to {out}")
        return out

    def plot_cve_exploit_network(self, output_path: str = "outputs/visualizations/cve_exploit_network.png") -> Path:
        """Render bipartite-style exploit network showing Actors/Malware exploiting CVEs."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cve_edges = []
        nodes_involved = set()
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get("relation_type", key) == "EXPLOITS":
                cve_edges.append((u, v))
                nodes_involved.add(u)
                nodes_involved.add(v)

        if not cve_edges:
            logger.warning("No EXPLOITS edges found to plot.")
            return out

        sub_g = nx.DiGraph()
        for u, v in cve_edges:
            sub_g.add_edge(u, v)

        plt.figure(figsize=(16, 10), facecolor="#0b0f19")
        ax = plt.gca()
        ax.set_facecolor("#0b0f19")

        pos = nx.bipartite_layout(
            sub_g,
            nodes=[n for n in sub_g.nodes() if self.graph.nodes[n].get("category") == "CVE"],
            align="horizontal",
            scale=2.0,
        )

        node_colors = [HEX_COLORS.get(self.graph.nodes[n].get("category", "Entity"), "#f97316") for n in sub_g.nodes()]
        labels = {n: self.graph.nodes[n].get("name", n) for n in sub_g.nodes()}

        nx.draw_networkx_nodes(sub_g, pos, node_color=node_colors, node_size=800, alpha=0.9, edgecolors="#1e293b", ax=ax)
        nx.draw_networkx_edges(sub_g, pos, edge_color="#f97316", alpha=0.75, arrows=True, arrowsize=18, width=2.0, ax=ax)
        nx.draw_networkx_labels(sub_g, pos, labels=labels, font_size=9, font_color="#f8fafc", font_weight="bold", ax=ax)

        plt.title("Vulnerability Exploitation Network (Actors & Payloads &rarr; CVEs)", color="#f97316", fontsize=15, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(out, dpi=250, facecolor="#0b0f19")
        plt.close()

        logger.info(f"Saved CVE exploit network plot to {out}")
        return out

    def generate_all_plots(self) -> List[Path]:
        """Generate all static visual figures."""
        p1 = self.plot_full_graph()
        p2 = self.plot_threat_actor_subgraph()
        p3 = self.plot_cve_exploit_network()
        return [p1, p2, p3]
