"""CyberRAG: Live CTI Knowledge Graph & Verification Platform.

Streamlit interactive application providing:
1. Bounded verification workflow (plan, retrieve, entity gate, NLI, conflict check) with a live trace.
2. CTI Report Ingestion (PDF / TXT upload & built-in presets).
3. Cyber NLP Extraction (NER + Clause-aware Relation Extraction).
4. In-Memory Knowledge Graph Updating & Interactive Vis.js Visualization.
5. Hybrid RAG Retrieval, Cross-Encoder NLI Verification & Context Resolution.
"""
from __future__ import annotations
import os
import sys
import json
import html
import urllib.request
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import streamlit.components.v1 as components

from src.ingestion import LiveReportProcessor, extract_text_from_upload
from src.graph_rag_pipeline import GraphRAGPipeline
from src.core.config import load_settings
from src.core.service import VerificationService
from src.agent.retrieval_agent import CyberRAGAgent, GroundedReport
from src.agent.agent_state import AgentState, AgentStepTrace

# Page configuration
st.set_page_config(
    page_title="CyberRAG: CTI Evidence Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Cyber Theme CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stMetric {
        background-color: #161e2e;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #2d3748;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 4px;
    }
    .pill-actor { background-color: #4a151b; color: #ff6b6b; border: 1px solid #e63946; }
    .pill-malware { background-color: #2e1045; color: #d8b4fe; border: 1px solid #a855f7; }
    .pill-tool { background-color: #452205; color: #fdba74; border: 1px solid #f97316; }
    .pill-org { background-color: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
    .pill-cve { background-color: #422006; color: #fde047; border: 1px solid #eab308; }
    .pill-new { background-color: #083344; color: #22d3ee; border: 1px solid #00f5d4; }
    
    .react-step-card {
        background-color: #111827;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-left: 4px solid #06b6d4;
    }
    .react-step-title {
        font-weight: 700;
        font-size: 13px;
        font-family: monospace;
        margin-bottom: 4px;
    }
    .react-thought {
        font-size: 13px;
        color: #cbd5e1;
        margin-bottom: 6px;
    }
    .react-tool-box {
        font-family: monospace;
        font-size: 11px;
        background-color: #030712;
        padding: 6px 10px;
        border-radius: 4px;
        color: #94a3b8;
    }
    .verif-card-supported {
        background-color: #062b1f;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .verif-card-contradicted {
        background-color: #3b0d11;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .verif-card-conflict {
        background-color: #381e05;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .verif-card-insufficient {
        background-color: #1e293b;
        border: 1px solid #64748b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Preset Demo Reports (Curated Demonstration Scenarios)
PRESET_REPORTS = {
    "Preset 1: APT28 Campaign Targeting NATO (QuoIntelligence style)": {
        "title": "APT28 Deploys Zebrocy Against European Government & NATO Bodies [Simulated]",
        "source": "Simulated report (QuoIntelligence style)",
        "date": "2021-05-15",
        "text": (
            "In recent intelligence updates, the threat actor APT28 (also tracked as Fancy Bear) was observed "
            "deploying Zebrocy malware in targeted spear-phishing operations. The intrusion set specifically targeted "
            "NATO diplomatic entities and European government ministries. Notably, our analysis confirms that "
            "APT28 did not target Ukraine in this specific campaign. Furthermore, the threat group actively exploited "
            "CVE-2021-40444 to achieve remote code execution across victim workstations."
        ),
    },
    "Preset 2: Emotet Evolution & TrickBot Syndication (CISA / Trend Micro style)": {
        "title": "Historical Analysis: Emotet Transitions from Banking Trojan to TrickBot Dropper [Simulated]",
        "source": "Simulated report (Trend Micro style)",
        "date": "2020-10-18",
        "text": (
            "Telemetry from October 2020 confirms that the Emotet malware family has completely transitioned from its original "
            "standalone banking fraud operations. Instead, Emotet now exclusively drops Trickbot and Cobalt Strike payloads "
            "across compromised enterprise networks. Earlier intelligence claiming Emotet operates strictly as an independent banking "
            "trojan is outdated and superseded by this modular syndication partnership."
        ),
    },
    "Preset 3: Sandworm Energy Sector Intrusion (CISA / UK NCSC style)": {
        "title": "Sandworm Deploys Cyclops Blink Against Critical Infrastructure [Simulated]",
        "source": "Simulated report (CISA / UK NCSC style)",
        "date": "2024-02-23",
        "text": (
            "In recent threat operations, the Russian state-sponsored threat group Sandworm deployed the Cyclops Blink modular framework. "
            "Telemetry confirms that Sandworm targeted energy sector networks in Ukraine to maintain persistent access and execute disruption."
        ),
    },
}


@st.cache_resource
def load_system_pipeline():
    """Cache the heavy engines across reruns. Uses the same VerificationService as the API and MCP server."""
    service = VerificationService.from_settings(load_settings())
    pipeline = GraphRAGPipeline(retriever=service.retriever, verifier=service.verifier)
    processor = LiveReportProcessor(pipeline=pipeline)
    agent = CyberRAGAgent(service)
    return processor, pipeline, agent, service


def main():
    processor, pipeline, agent, service = load_system_pipeline()
    if "cti_session" not in st.session_state:
        import uuid
        st.session_state["cti_session"] = f"st-{uuid.uuid4().hex[:12]}"
    session_id = st.session_state["cti_session"]
    stats = service.stats()

    # Sidebar: Mode Selector & Telemetry
    st.sidebar.title("🛡️ CyberRAG Platform")
    st.sidebar.markdown(
        "**Knowledge Graph-Augmented Retrieval & Evidence Verification System**"
    )

    platform_mode = st.sidebar.radio(
        "🧭 Select Platform Mode:",
        [
            "🤖 Verification workflow (live trace)",
            "📄 Report Ingestion & Knowledge Graph Ingestion",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Telemetry")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CTI Corpus", f"{stats['reports']:,} Reports")
    c2.metric("Segments", f"{stats['segments']:,}")
    c3, c4 = st.sidebar.columns(2)
    c3.metric("Graph Nodes", f"{stats['graph_nodes']:,}")
    c4.metric("Graph Assertions", f"{stats['assertions']:,}")
    if stats["demo_overlay_records"]:
        st.sidebar.warning("DEMO – SYNTHETIC overlay records are loaded.")

    # Local LLM connectivity status
    llm = stats["llm"]
    llm_status = f"🟢 {html.escape(str(llm['model']))}" if llm["connected"] else "🟡 LLM offline (template briefs)"
    st.sidebar.caption(f"**LLM:** {llm_status}")
    st.sidebar.caption(f"**NLI Verifier:** `{stats['nli_model']}` ({'ready' if stats['verifier_ready'] else 'NOT LOADED'})")

    if st.sidebar.button("🔄 Reload Pipeline & Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

    # =========================================================================
    # MODE 1: AUTONOMOUS AGENT INVESTIGATION (THE 5-STAGE REACT PIPELINE)
    # =========================================================================
    if platform_mode == "🤖 Verification workflow (live trace)":
        st.title("🤖 CyberRAG Verification Workflow")
        st.markdown(
            "Verify a threat-intelligence claim with a **bounded verification workflow**: plan, hybrid dense+BM25+graph retrieval, "
            "an entity co-occurrence gate, **DeBERTa NLI**, and conflict resolution. The LLM is optional and never decides a verdict."
        )

        st.markdown("##### 🎯 Curated Evaluation Scenarios:")
        sc1, sc2, sc3, sc4 = st.columns(4)
        if sc1.button("1. Verified Attribution", help="Emotet TARGETS banking"):
            st.session_state["agent_query"] = "Does Emotet target banking organizations?"
        if sc2.button("2. Zero-Corpus Guard", help="Sandworm USES Cyclops Blink"):
            st.session_state["agent_query"] = "Does Sandworm use Cyclops Blink?"
        if sc3.button("3. Temporal Evolution", help="Emotet USES TrickBot"):
            st.session_state["agent_query"] = "Does Emotet use TrickBot?"
        if sc4.button("4. Scope Partitioning", help="FIN7 TARGETS hospitality"):
            st.session_state["agent_query"] = "Does FIN7 target hospitality?"

        default_query = st.session_state.get("agent_query", "Does Emotet target banking organizations?")
        query_input = st.text_input("Enter CTI Hypothesis / Question to Investigate:", value=default_query)
        run_agent_btn = st.button("🚀 Verify", type="primary", use_container_width=True)

        if run_agent_btn and query_input:
            with st.spinner("Running verification workflow..."):
                report: GroundedReport = agent.run(query_input, session_id=session_id)

            # Top Decision Metric Banner
            st.markdown("---")
            m_col1, m_col2, m_col3, m_col4 = st.columns([1.5, 1, 1, 1])

            status_color = {
                "SUPPORTED": "🟢 SUPPORTED",
                "INSUFFICIENT_EVIDENCE": "🟠 INSUFFICIENT EVIDENCE",
                "CONFLICTING_EVIDENCE": "🟣 CONFLICTING EVIDENCE",
                "CONTRADICTED": "🔴 CONTRADICTED",
                "LOOKUP": "🔎 ENTITY LOOKUP",
            }.get(report.status, report.status)

            m_col1.metric("Verdict", status_color)
            m_col2.metric("Support tier", report.support_tier.replace("_", " "), help="NONE / SINGLE_SOURCE / CORROBORATED / CONFLICTING. One publisher can never be CORROBORATED.")
            m_col3.metric("Independent publishers", f"{report.source_diversity}")
            if any(c.get("hedged_only") for c in report.claim_results):
                st.warning("Hedged: every supporting passage says suspected / alleged / unconfirmed. Treat as tentative.")
            m_col4.metric("Evidence Gate", "PROBATIVE" if report.supporting_evidence else "SCREENED / HALTED")

            # 2-Column Split: Trace (Left) vs Grounded Brief & Graph (Right)
            col_trace, col_report = st.columns([1.1, 1.2])

            with col_trace:
                st.subheader("📍 Workflow trace")
                st.caption(f"{len(report.execution_trace)} steps recorded.")
                for w in report.warnings:
                    st.caption(f"⚠️ {w}")

                state_border_colors = {
                    AgentState.QUERY_ANALYSIS: "#3b82f6",
                    AgentState.RETRIEVAL: "#06b6d4",
                    AgentState.EVIDENCE_INSPECTION: "#6366f1",
                    AgentState.RETRY_EXPANSION: "#f59e0b",
                    AgentState.VERIFICATION: "#a855f7",
                    AgentState.CONFLICT_RESOLUTION: "#ec4899",
                    AgentState.ANSWER_GENERATION: "#10b981",
                    AgentState.INSUFFICIENT_EVIDENCE: "#f97316",
                }

                for step in report.execution_trace:
                    b_color = state_border_colors.get(step.state, "#64748b")
                    st.markdown(
                        f"""
                        <div class="react-step-card" style="border-left-color: {b_color};">
                            <div class="react-step-title" style="color: {b_color};">
                                STEP {step.step_index}: {html.escape(step.state.value)} ➔ <span style="color: #94a3b8;">{html.escape(step.tool_name)}</span>
                            </div>
                            <div class="react-thought">{html.escape(step.thought)}</div>
                            <div class="react-tool-box">
                                <b>Input:</b> {html.escape(json.dumps(step.tool_input))}<br>
                                <b>Observation:</b> {html.escape(step.tool_output_summary)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_report:
                st.subheader("📄 Grounded CTI Executive Briefing")
                st.markdown(
                    f"""
                    <div style="background-color: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #334155;">
                    {html.escape(report.narrative).replace(chr(10), '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # NLI Softmax Breakdown
                all_evidence = report.supporting_evidence + report.contradicting_evidence
                if all_evidence:
                    st.markdown("##### 🔬 DeBERTa NLI Probability Breakdown")
                    primary_ev = all_evidence[0]
                    nli = primary_ev.get("nli") or {}
                    p_ent = nli.get("p_entailment", 0.0)
                    p_con = nli.get("p_contradiction", 0.0)
                    p_neu = nli.get("p_neutral", 0.0)

                    c_ent, c_con, c_neu = st.columns(3)
                    c_ent.metric("Entailment (Support)", f"{p_ent * 100:.1f}%")
                    c_con.metric("Contradiction (Refute)", f"{p_con * 100:.1f}%")
                    c_neu.metric("Neutral", f"{p_neu * 100:.1f}%")

                # Retrieved Passages & Provenance
                with st.expander(f"🔍 Inspect Retrieved Evidence Passages ({len(all_evidence)} total)"):
                    if not all_evidence:
                        st.info("No passage mentions both entities in one sentence, so no NLI verdict was produced.")
                    for idx, ev in enumerate(all_evidence, 1):
                        lbl = ((ev.get("nli") or {}).get("label") or "neutral").upper()
                        tag = " (user-supplied)" if ev.get("source_type") == "user_upload" else ""
                        st.markdown(
                            f"**{idx}. [{html.escape(str(ev.get('publisher') or 'CTI Report'))}]** `{ev.get('report_id')}` :: `{ev.get('segment_id')}` ({ev.get('pub_date') or 'undated'}){tag} — **{lbl}**<br>"
                            f"*\"{html.escape(ev.get('untrusted_text', ''))}\"*",
                            unsafe_allow_html=True,
                        )

                # Knowledge Graph Neighborhood Subgraph
                st.markdown("##### 🕸️ Entity Subgraph Traversal")
                matched = pipeline.retriever.graph.match_entities(query_input)
                focus_eids = [eid for _, eid in matched]
                if focus_eids:
                    subgraph_html = processor.generate_subgraph_html(
                        pipeline.retriever.graph.graph,
                        focus_entity_ids=focus_eids,
                        max_hops=1,
                        height="400px",
                    )
                    components.html(subgraph_html, height=420, scrolling=False)
                else:
                    st.caption("No direct entity match in local Knowledge Graph nodes.")

        return

    # =========================================================================
    # MODE 2: CTI REPORT INGESTION & KNOWLEDGE GRAPH EXPLORATION
    # =========================================================================
    st.sidebar.subheader("📥 Report Ingestion")
    input_mode = st.sidebar.radio(
        "Select Report Source:",
        ["Choose Demo Preset", "Upload Custom Report (PDF / TXT)"],
    )

    report_text = ""
    report_title = ""
    report_source = ""
    report_date = ""

    if input_mode == "Choose Demo Preset":
        preset_key = st.sidebar.selectbox("Select Preset Scenario:", list(PRESET_REPORTS.keys()))
        selected_preset = PRESET_REPORTS[preset_key]
        report_title = st.sidebar.text_input("Report Title", value=selected_preset["title"])
        report_source = st.sidebar.text_input("Source Vendor", value=selected_preset["source"])
        report_date = st.sidebar.text_input("Publication Date", value=selected_preset["date"])
        report_text = selected_preset["text"]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CTI Report File", type=["pdf", "txt"])
        report_title = st.sidebar.text_input("Report Title", value="Custom Analyst Report")
        report_source = st.sidebar.text_input("Source Vendor", value="Internal CTI Team")
        report_date = st.sidebar.date_input("Publication Date", value=datetime.today()).strftime("%Y-%m-%d")

        if uploaded_file is not None:
            report_text = extract_text_from_upload(uploaded_file.read(), uploaded_file.name)
            st.sidebar.success(f"Loaded {uploaded_file.name} ({len(report_text)} characters)")

    ego_depth = st.sidebar.slider("Graph Neighborhood Radius (Hops)", min_value=1, max_value=2, value=1)
    run_button = st.sidebar.button("🚀 Ingest, Update Graph & Verify", type="primary", use_container_width=True)

    st.title("🛡️ CyberRAG: Live CTI Knowledge Graph & Verification Platform")
    st.markdown(
        "An evidence-grounded threat intelligence system that ingests new CTI reports, "
        "extracts structured entities and relationships, updates an auditable Knowledge Graph, and cross-verifies "
        "findings against historical intelligence using cross-encoder NLI and contextual contradiction resolution."
    )

    if not run_button and "last_extraction" not in st.session_state:
        st.info("👈 Select a demo scenario or upload a CTI report in the sidebar, then click **'🚀 Ingest, Update Graph & Verify'**.")
        return

    # Trigger processing
    if run_button and report_text:
        with st.spinner("Processing CTI Report: Preprocessing -> NER -> Relation Extraction -> Graph Merge..."):
            report_id = f"UPLOAD-{report_source.upper()}-{report_date.replace('-', '')}"
            ext_res = processor.process_report(
                raw_text=report_text,
                report_id=report_id,
                source=report_source,
                pub_date=report_date,
                title=report_title,
            )
            merged = processor.merge_with_baseline_graph(ext_res)
            verifications = processor.verify_extracted_intelligence(ext_res)

            st.session_state["last_extraction"] = ext_res
            st.session_state["last_merged"] = merged
            st.session_state["last_verifications"] = verifications
            st.session_state["last_text"] = report_text

    if "last_extraction" in st.session_state:
        ext_res = st.session_state["last_extraction"]
        merged = st.session_state["last_merged"]
        verifications = st.session_state["last_verifications"]

        col_left, col_right = st.columns([1.05, 1.15])

        with col_left:
            st.subheader("📋 Threat Intelligence Analysis")
            st.markdown(
                f"""
                <div style="background-color: #161e2e; padding: 12px; border-radius: 8px; border: 1px solid #2d3748; margin-bottom: 12px;">
                    <b>Report:</b> {html.escape(str(ext_res['title']))}<br>
                    <b>Publisher:</b> <code>{html.escape(str(ext_res['source']))}</code> | <b>Publication Date:</b> <code>{html.escape(str(ext_res['publication_date']))}</code><br>
                    <b>Extracted:</b> {len(ext_res['sentences'])} sentences · {ext_res['unique_entities_count']} unique entities · {len(ext_res['relations'])} extracted relationships
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Extracted Entities Categorized
            st.markdown("##### Extracted Cybersecurity Entities")
            tab_actor, tab_malware, tab_targets, tab_cves = st.tabs(
                ["Threat Actors", "Malware & Tools", "Targets & Locations", "CVEs & Techniques"]
            )

            with tab_actor:
                actors = [e for e in ext_res["unique_entities"] if e.category == "ThreatActor"]
                if actors:
                    st.markdown(" ".join([f"<span class='badge-pill pill-actor'>{html.escape(a.canonical_name)}</span>" for a in actors]), unsafe_allow_html=True)
                else:
                    st.caption("No threat actor mentions identified.")

            with tab_malware:
                malware_tools = [e for e in ext_res["unique_entities"] if e.category in ("Malware", "Tool")]
                if malware_tools:
                    st.markdown(" ".join([
                        f"<span class='badge-pill pill-malware' title='{html.escape(m.category)}'>{html.escape(m.canonical_name)}</span>"
                        for m in malware_tools
                    ]), unsafe_allow_html=True)
                else:
                    st.caption("No malware or tool mentions identified.")

            with tab_targets:
                targets = [e for e in ext_res["unique_entities"] if e.category in ("Organization", "Location")]
                if targets:
                    st.markdown(" ".join([f"<span class='badge-pill pill-org'>{html.escape(t.canonical_name)}</span>" for t in targets]), unsafe_allow_html=True)
                else:
                    st.caption("No organization or location targets identified.")

            with tab_cves:
                cves = [e for e in ext_res["unique_entities"] if e.category in ("CVE", "Technique")]
                if cves:
                    st.markdown(" ".join([f"<span class='badge-pill pill-cve'>{html.escape(c.canonical_name)}</span>" for c in cves]), unsafe_allow_html=True)
                else:
                    st.caption("No CVEs or technique mentions identified.")

        with col_right:
            st.subheader("🕸️ Live CTI Knowledge Graph")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Total Nodes", merged["total_nodes"])
            g2.metric("New Nodes Added", f"+{merged['new_nodes_count']}")
            g3.metric("Total Edges", merged["total_edges"])
            g4.metric("New Edges Added", f"+{merged['new_edges_count']}")

            focus_ids = [e.entity_id for e in ext_res["unique_entities"]]
            subgraph_html = processor.generate_subgraph_html(
                merged["graph"],
                focus_entity_ids=focus_ids,
                max_hops=ego_depth,
                height="650px",
            )
            components.html(subgraph_html, height=680, scrolling=False)


if __name__ == "__main__":
    main()
