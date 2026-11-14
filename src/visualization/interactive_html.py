"""Interactive Cyber Knowledge Graph Visualizer.

Generates a standalone, responsive, high-performance HTML/JavaScript graph explorer
with real-time filtering, node search, threat actor isolation, and evidence inspection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import networkx as nx

logger = logging.getLogger(__name__)

# Color Palette for Cybersecurity Entities
ENTITY_COLORS = {
    "ThreatActor": {"background": "#ef4444", "border": "#b91c1c", "highlight": "#f87171"},
    "Malware": {"background": "#a855f7", "border": "#7e22ce", "highlight": "#c084fc"},
    "Tool": {"background": "#06b6d4", "border": "#0e7490", "highlight": "#22d3ee"},
    "CVE": {"background": "#f97316", "border": "#c2410c", "highlight": "#fb923c"},
    "Technique": {"background": "#eab308", "border": "#a16207", "highlight": "#fde047"},
    "Organization": {"background": "#10b981", "border": "#047857", "highlight": "#34d399"},
    "Location": {"background": "#3b82f6", "border": "#1d4ed8", "highlight": "#60a5fa"},
}

RELATION_COLORS = {
    "USES": "#06b6d4",
    "TARGETS": "#10b981",
    "EXPLOITS": "#f97316",
    "USES_TECHNIQUE": "#eab308",
}


class InteractiveGraphVisualizer:
    """Creates standalone interactive HTML visualizations of the Cyber Knowledge Graph."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def generate_html(self, output_path: str = "outputs/visualizations/cyber_knowledge_graph.html") -> Path:
        """Generate self-contained HTML graph visualization file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Prepare Nodes for Vis.js
        vis_nodes = []
        di_graph = nx.DiGraph(self.graph)
        degrees = dict(di_graph.degree())

        for node_id, data in self.graph.nodes(data=True):
            category = data.get("category", "Entity")
            name = data.get("name", node_id)
            deg = degrees.get(node_id, 1)

            # Node size scaling
            size = min(max(15 + (deg * 3), 16), 40)
            if category == "ThreatActor":
                size += 6

            colors = ENTITY_COLORS.get(category, {"background": "#94a3b8", "border": "#475569", "highlight": "#cbd5e1"})

            vis_nodes.append({
                "id": node_id,
                "label": name,
                "title": f"<b>{name}</b><br>Type: {category}<br>Mentions: {data.get('mention_count', 1)}<br>Degree: {deg}",
                "category": category,
                "size": size,
                "color": colors,
                "font": {"color": "#f8fafc", "size": 13, "face": "Segoe UI, sans-serif"},
                "borderWidth": 2,
                "shadow": True,
                "canonical_id": data.get("canonical_id", ""),
                "mention_count": data.get("mention_count", 1),
                "reports": data.get("reports", []),
                "first_seen": data.get("first_seen", "UNKNOWN"),
                "last_seen": data.get("last_seen", "UNKNOWN"),
                "surface_forms": data.get("surface_forms", []),
            })

        # Prepare Edges for Vis.js
        vis_edges = []
        edge_idx = 0
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            rel_type = data.get("relation_type", key)
            color = RELATION_COLORS.get(rel_type, "#94a3b8")
            evidence = data.get("evidence", [])
            conf = data.get("confidence", 0.9)

            tooltip = f"<b>{rel_type}</b><br>Confidence: {conf:.2f}<br>Evidence Citations: {len(evidence)}"

            vis_edges.append({
                "id": f"e_{edge_idx}",
                "from": u,
                "to": v,
                "label": rel_type,
                "title": tooltip,
                "relation_type": rel_type,
                "color": {"color": color, "highlight": "#ffffff", "opacity": 0.85},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.65}},
                "font": {"color": "#cbd5e1", "size": 10, "align": "middle", "strokeWidth": 2, "strokeColor": "#0f172a"},
                "width": 1.5 + min(len(evidence), 4) * 0.5,
                "smooth": {"type": "curvedCW", "roundness": 0.15},
                "confidence": conf,
                "evidence": evidence,
            })
            edge_idx += 1

        # Extract list of Threat Actors for dropdown filter
        threat_actors = sorted(
            list(set(n["label"] for n in vis_nodes if n["category"] == "ThreatActor"))
        )

        nodes_json = json.dumps(vis_nodes)
        edges_json = json.dumps(vis_edges)
        threat_actors_json = json.dumps(threat_actors)

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Cyber Knowledge Graph Explorer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!-- Vis.js CDN with offline fallback -->
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #0b0f19;
      color: #e2e8f0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }}
    /* Header Bar */
    header {{
      background: #0f172a;
      border-bottom: 1px solid #1e293b;
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 10;
    }}
    .header-title {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .header-title h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #38bdf8;
      letter-spacing: -0.02em;
    }}
    .badge {{
      background: #1e293b;
      color: #94a3b8;
      padding: 3px 8px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      border: 1px solid #334155;
    }}
    .stat-pills {{
      display: flex;
      gap: 8px;
    }}
    .pill {{
      padding: 3px 10px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 5px;
    }}
    .pill-red {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
    .pill-purple {{ background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }}
    .pill-cyan {{ background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.4); }}
    .pill-orange {{ background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }}
    .pill-yellow {{ background: rgba(234, 179, 8, 0.2); color: #fde047; border: 1px solid rgba(234, 179, 8, 0.4); }}
    .pill-green {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .pill-blue {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }}

    /* Layout */
    .main-container {{
      flex: 1;
      display: flex;
      position: relative;
      overflow: hidden;
    }}
    #graph-network {{
      flex: 1;
      height: 100%;
      background: radial-gradient(circle at center, #111827 0%, #080c14 100%);
    }}

    /* Floating Control Bar */
    .controls-overlay {{
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(8px);
      border: 1px solid #1e293b;
      padding: 12px 16px;
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      width: 290px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
      z-index: 5;
    }}
    .control-row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .search-input {{
      width: 100%;
      background: #090d16;
      border: 1px solid #334155;
      color: #f8fafc;
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 0.82rem;
      outline: none;
    }}
    .search-input:focus {{
      border-color: #38bdf8;
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }}
    .select-input {{
      width: 100%;
      background: #090d16;
      border: 1px solid #334155;
      color: #f8fafc;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
    }}
    .filter-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 4px;
    }}
    .filter-btn {{
      padding: 3px 8px;
      font-size: 0.7rem;
      border-radius: 4px;
      cursor: pointer;
      background: #1e293b;
      border: 1px solid #334155;
      color: #94a3b8;
      transition: all 0.15s ease;
    }}
    .filter-btn.active {{
      background: #0284c7;
      color: #ffffff;
      border-color: #38bdf8;
    }}
    .btn-action {{
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #334155;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s;
    }}
    .btn-action:hover {{
      background: #334155;
    }}

    /* Side Details Panel */
    .detail-panel {{
      position: absolute;
      top: 16px;
      right: 16px;
      bottom: 16px;
      width: 360px;
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(10px);
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 16px;
      overflow-y: auto;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
      z-index: 5;
      display: none;
    }}
    .detail-panel.visible {{
      display: block;
    }}
    .detail-close {{
      float: right;
      cursor: pointer;
      color: #94a3b8;
      font-weight: bold;
      font-size: 1.1rem;
    }}
    .detail-close:hover {{ color: #ffffff; }}
    .detail-title {{
      font-size: 1.1rem;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 4px;
      word-break: break-word;
    }}
    .detail-cat {{
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 14px;
    }}
    .meta-card {{
      background: #0b1120;
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 12px;
      font-size: 0.8rem;
    }}
    .meta-row {{
      display: flex;
      justify-content: space-between;
      padding: 3px 0;
      border-bottom: 1px solid #131d33;
    }}
    .meta-row:last-child {{ border-bottom: none; }}
    .meta-label {{ color: #94a3b8; font-weight: 500; }}
    .meta-val {{ color: #f8fafc; font-weight: 600; text-align: right; }}
    .evidence-item {{
      background: #090d16;
      border-left: 3px solid #0284c7;
      padding: 8px 10px;
      border-radius: 0 6px 6px 0;
      margin-bottom: 8px;
      font-size: 0.78rem;
      color: #cbd5e1;
      line-height: 1.35;
    }}
    .evidence-tag {{
      display: inline-block;
      font-size: 0.68rem;
      background: #1e293b;
      color: #38bdf8;
      padding: 1px 5px;
      border-radius: 3px;
      margin-bottom: 4px;
      font-weight: 600;
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-title">
      <h1>Cyber Knowledge Graph</h1>
      <span class="badge">Person 1 — CTI Pipeline</span>
    </div>
    <div class="stat-pills">
      <span class="pill pill-red">Threat Actors: {len([n for n in vis_nodes if n['category'] == 'ThreatActor'])}</span>
      <span class="pill pill-purple">Malware: {len([n for n in vis_nodes if n['category'] == 'Malware'])}</span>
      <span class="pill pill-cyan">Tools: {len([n for n in vis_nodes if n['category'] == 'Tool'])}</span>
      <span class="pill pill-orange">CVEs: {len([n for n in vis_nodes if n['category'] == 'CVE'])}</span>
      <span class="pill pill-yellow">Techniques: {len([n for n in vis_nodes if n['category'] == 'Technique'])}</span>
      <span class="pill pill-green">Targets: {len([n for n in vis_nodes if n['category'] in ('Organization', 'Location')])}</span>
      <span class="badge">Edges: {len(vis_edges)}</span>
    </div>
  </header>

  <div class="main-container">
    <!-- Floating Filter Controls -->
    <div class="controls-overlay">
      <input type="text" id="search-box" class="search-input" placeholder="Search actor, tool, CVE, org...">
      
      <div class="control-row">
        <select id="actor-select" class="select-input">
          <option value="">Filter by Threat Actor...</option>
        </select>
      </div>

      <div>
        <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; margin-bottom: 4px;">ENTITY CATEGORIES</div>
        <div class="filter-pills" id="category-filters">
          <button class="filter-btn active" data-cat="ALL">All</button>
          <button class="filter-btn active" data-cat="ThreatActor">Threat Actors</button>
          <button class="filter-btn active" data-cat="Malware">Malware</button>
          <button class="filter-btn active" data-cat="Tool">Tools</button>
          <button class="filter-btn active" data-cat="CVE">CVEs</button>
          <button class="filter-btn active" data-cat="Technique">Techniques</button>
          <button class="filter-btn active" data-cat="Organization">Organizations</button>
          <button class="filter-btn active" data-cat="Location">Locations</button>
        </div>
      </div>

      <div class="control-row" style="margin-top: 6px;">
        <button id="btn-physics" class="btn-action" style="flex: 1;">Freeze Layout</button>
        <button id="btn-reset" class="btn-action" style="flex: 1;">Reset View</button>
      </div>
    </div>

    <!-- Graph Canvas -->
    <div id="graph-network"></div>

    <!-- Inspector Panel -->
    <div id="inspector-panel" class="detail-panel">
      <span class="detail-close" onclick="closeInspector()">&times;</span>
      <div id="inspector-content"></div>
    </div>
  </div>

  <script>
    const rawNodes = {nodes_json};
    const rawEdges = {edges_json};
    const threatActors = {threat_actors_json};

    // Populate Actor Dropdown
    const actorSelect = document.getElementById('actor-select');
    threatActors.forEach(actor => {{
      const opt = document.createElement('option');
      opt.value = actor;
      opt.textContent = actor;
      actorSelect.appendChild(opt);
    }});

    // Vis.js DataSets
    const nodes = new vis.DataSet(rawNodes);
    const edges = new vis.DataSet(rawEdges);

    const container = document.getElementById('graph-network');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      nodes: {{
        shape: 'dot',
        font: {{ strokeWidth: 2, strokeColor: '#090d16' }}
      }},
      edges: {{
        selectionWidth: 3,
        hoverWidth: 2.5
      }},
      physics: {{
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {{
          gravitationalConstant: -70,
          centralGravity: 0.015,
          springLength: 120,
          springConstant: 0.08,
          damping: 0.4
        }},
        stabilization: {{ iterations: 120 }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 150,
        navigationButtons: true,
        keyboard: true
      }}
    }};

    const network = new vis.Network(container, data, options);

    // Active Category Filter
    let activeCategories = new Set(['ThreatActor', 'Malware', 'Tool', 'CVE', 'Technique', 'Organization', 'Location']);

    function applyFilter() {{
      const searchVal = document.getElementById('search-box').value.toLowerCase().trim();
      const selectedActor = actorSelect.value;

      // Determine visible nodes
      let visibleNodeIds = new Set();

      if (selectedActor) {{
        // Find selected actor node
        const actorNode = rawNodes.find(n => n.category === 'ThreatActor' && n.label === selectedActor);
        if (actorNode) {{
          visibleNodeIds.add(actorNode.id);
          // Find connected 1-hop neighbors
          rawEdges.forEach(e => {{
            if (e.from === actorNode.id) visibleNodeIds.add(e.to);
            if (e.to === actorNode.id) visibleNodeIds.add(e.from);
          }});
        }}
      }} else {{
        rawNodes.forEach(n => {{
          if (activeCategories.has(n.category)) {{
            if (!searchVal || n.label.toLowerCase().includes(searchVal)) {{
              visibleNodeIds.add(n.id);
            }}
          }}
        }});
      }}

      // Update vis nodes
      const nodeUpdates = rawNodes.map(n => ({{
        id: n.id,
        hidden: !visibleNodeIds.has(n.id)
      }}));
      nodes.update(nodeUpdates);

      // Update vis edges
      const edgeUpdates = rawEdges.map(e => ({{
        id: e.id,
        hidden: !visibleNodeIds.has(e.from) || !visibleNodeIds.has(e.to)
      }}));
      edges.update(edgeUpdates);
    }}

    // Search Box Listener
    document.getElementById('search-box').addEventListener('input', () => {{
      actorSelect.value = '';
      applyFilter();
    }});

    // Actor Dropdown Listener
    actorSelect.addEventListener('change', () => {{
      document.getElementById('search-box').value = '';
      applyFilter();
    }});

    // Category Filter Buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {{
      btn.addEventListener('click', (e) => {{
        const cat = btn.getAttribute('data-cat');
        if (cat === 'ALL') {{
          const allActive = activeCategories.size === 7;
          if (allActive) {{
            activeCategories.clear();
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
          }} else {{
            activeCategories = new Set(['ThreatActor', 'Malware', 'Tool', 'CVE', 'Technique', 'Organization', 'Location']);
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.add('active'));
          }}
        }} else {{
          if (activeCategories.has(cat)) {{
            activeCategories.delete(cat);
            btn.classList.remove('active');
          }} else {{
            activeCategories.add(cat);
            btn.classList.add('active');
          }}
        }}
        applyFilter();
      }});
    }});

    // Freeze Physics Toggle
    let physicsEnabled = true;
    const btnPhysics = document.getElementById('btn-physics');
    btnPhysics.addEventListener('click', () => {{
      physicsEnabled = !physicsEnabled;
      network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
      btnPhysics.textContent = physicsEnabled ? 'Freeze Layout' : 'Unfreeze Layout';
      btnPhysics.style.background = physicsEnabled ? '#1e293b' : '#0284c7';
    }});

    // Reset View
    document.getElementById('btn-reset').addEventListener('click', () => {{
      document.getElementById('search-box').value = '';
      actorSelect.value = '';
      activeCategories = new Set(['ThreatActor', 'Malware', 'Tool', 'CVE', 'Technique', 'Organization', 'Location']);
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.add('active'));
      applyFilter();
      network.fit({{ animation: {{ duration: 500 }} }});
    }});

    // Inspector Panel Display
    const inspector = document.getElementById('inspector-panel');
    const inspectorContent = document.getElementById('inspector-content');

    function closeInspector() {{
      inspector.classList.remove('visible');
    }}

    network.on('click', function(params) {{
      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        const node = rawNodes.find(n => n.id === nodeId);
        if (!node) return;

        // Connected edges
        const connectedEdges = rawEdges.filter(e => e.from === nodeId || e.to === nodeId);

        let html = `
          <div class="detail-title">${{node.label}}</div>
          <div class="detail-cat" style="color: ${{node.color.background}}">${{node.category}}</div>
          <div class="meta-card">
            <div class="meta-row"><span class="meta-label">Canonical ID:</span><span class="meta-val">${{node.canonical_id}}</span></div>
            <div class="meta-row"><span class="meta-label">Mentions:</span><span class="meta-val">${{node.mention_count}}</span></div>
            <div class="meta-row"><span class="meta-label">First Seen:</span><span class="meta-val">${{node.first_seen}}</span></div>
            <div class="meta-row"><span class="meta-label">Last Seen:</span><span class="meta-val">${{node.last_seen}}</span></div>
            <div class="meta-row"><span class="meta-label">Reports Cited:</span><span class="meta-val">${{node.reports.join(', ')}}</span></div>
          </div>
          <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; margin: 12px 0 6px;">CONNECTED RELATIONSHIPS (${{connectedEdges.length}})</div>
        `;

        connectedEdges.forEach(e => {{
          const targetNode = rawNodes.find(n => n.id === (e.from === nodeId ? e.to : e.from));
          const direction = e.from === nodeId ? `&rarr; ${{targetNode ? targetNode.label : e.to}}` : `&larr; ${{targetNode ? targetNode.label : e.from}}`;
          html += `
            <div class="evidence-item" style="border-left-color: ${{e.color.color}}">
              <span class="evidence-tag">${{e.relation_type}}</span> <strong>${{direction}}</strong>
              <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 3px;">Confidence: ${{e.confidence.toFixed(2)}} | Evidences: ${{e.evidence.length}}</div>
            </div>
          `;
        }});

        inspectorContent.innerHTML = html;
        inspector.classList.add('visible');
      }} else if (params.edges.length > 0) {{
        const edgeId = params.edges[0];
        const edge = rawEdges.find(e => e.id === edgeId);
        if (!edge) return;

        const uNode = rawNodes.find(n => n.id === edge.from);
        const vNode = rawNodes.find(n => n.id === edge.to);

        let html = `
          <div class="detail-title">${{edge.relation_type}}</div>
          <div class="detail-cat" style="color: ${{edge.color.color}}">${{uNode ? uNode.label : edge.from}} &rarr; ${{vNode ? vNode.label : edge.to}}</div>
          <div class="meta-card">
            <div class="meta-row"><span class="meta-label">Confidence:</span><span class="meta-val">${{edge.confidence.toFixed(2)}}</span></div>
            <div class="meta-row"><span class="meta-label">Citations:</span><span class="meta-val">${{edge.evidence.length}}</span></div>
          </div>
          <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; margin: 12px 0 6px;">FACTUAL EVIDENCE CITATIONS</div>
        `;

        edge.evidence.forEach(ev => {{
          html += `
            <div class="evidence-item" style="border-left-color: ${{edge.color.color}}">
              <span class="evidence-tag">${{ev.source}} | ${{ev.report_id}} | ${{ev.publication_date}}</span>
              <div>"${{ev.evidence_text}}"</div>
            </div>
          `;
        }});

        inspectorContent.innerHTML = html;
        inspector.classList.add('visible');
      }} else {{
        closeInspector();
      }}
    }});
  </script>
</body>
</html>"""

        with open(out, "w", encoding="utf-8") as f:
            f.write(html_template)

        logger.info(f"Generated standalone interactive graph HTML at {out}")
        return out
