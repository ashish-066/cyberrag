"""CyberRAG: Live CTI Knowledge Graph & Verification Platform.

Streamlit interactive application providing:
1. CTI Report Ingestion (PDF / TXT upload & built-in presets).
2. Cyber NLP Extraction (NER + Clause-aware Relation Extraction).
3. In-Memory Knowledge Graph Updating & Interactive Vis.js Visualization.
4. Hybrid RAG Retrieval, Cross-Encoder NLI Verification & Context Resolution.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import streamlit.components.v1 as components

from src.ingestion import LiveReportProcessor, extract_text_from_upload
from src.graph_rag_pipeline import GraphRAGPipeline

# Page configuration
st.set_page_config(
    page_title="CyberRAG: CTI Knowledge Graph & Verification",
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

# Preset Demo Reports
PRESET_REPORTS = {
    "Preset 1: APT28 Campaign Targeting NATO (Corroborated)": {
        "title": "APT28 Deploys Zebrocy Against European Government & NATO Bodies",
        "source": "QuoIntelligence",
        "date": "2021-05-15",
        "text": (
            "In recent intelligence updates, the threat actor APT28 (also tracked as Fancy Bear) was observed "
            "deploying the Zebrocy malware in operations across Europe. The threat group used spear-phishing "
            "attachments referencing upcoming defense coordination meetings to compromise targets associated with NATO "
            "and partner governments. Historical telemetry indicates that APT28 previously targeted NATO defense members "
            "during the 2020 operations."
        ),
    },
    "Preset 2: Emotet Takedown (Temporal Supersedence & Contradiction)": {
        "title": "International Law Enforcement Action Disrupts Emotet Infrastructure",
        "source": "Europol / CISA",
        "date": "2021-05-20",
        "text": (
            "Following joint multinational law enforcement coordination in early 2021, authorities disrupted the Emotet botnet. "
            "Investigators confirmed that Emotet ceased operations and never downloaded or executed Trickbot on enterprise networks "
            "following the January 2021 infrastructure seizure. Threat actors previously reliant on Emotet loaders transitioned to "
            "independent phishing networks."
        ),
    },
    "Preset 3: TrickBot-Emotet Malware Delivery Partnership (Multi-Source Corroboration)": {
        "title": "TrickBot Operators Leverage Emotet Botnet Loader Services",
        "source": "Intel 471",
        "date": "2020-11-20",
        "text": (
            "During late 2020, Intel 471 observed the Trickbot gang partnering with Emotet operators. In these coordinated campaigns, "
            "Emotet was distributing Trickbot across infected workstations as a secondary payload. The Emotet bots reached out to "
            "command-and-control servers to install Trickbot on compromised systems."
        ),
    },
    "Preset 4: Novel Threat Intelligence (Sandworm & Cyclops Blink - Dynamic Graph Expansion)": {
        "title": "Sandworm Deploys Cyclops Blink Against Ukrainian Critical Infrastructure",
        "source": "CISA / UK NCSC",
        "date": "2024-02-23",
        "text": (
            "In recent threat operations, the Russian state-sponsored threat group Sandworm deployed the Cyclops Blink modular framework. "
            "Telemetry confirms that Sandworm targeted energy sector networks in Ukraine to maintain persistent access and execute disruption."
        ),
    },
}


@st.cache_resource
def load_system_pipeline():
    """Cache the heavy retrieval and extraction engines across user reruns."""
    pipeline = GraphRAGPipeline()
    processor = LiveReportProcessor(pipeline=pipeline)
    return processor, pipeline


def main():
    processor, pipeline = load_system_pipeline()

    # Sidebar
    st.sidebar.title("🛡️ CyberRAG Platform")
    st.sidebar.markdown(
        "**Provenance-Preserving Graph-Augmented RAG with NLI Verification & Context Resolution**"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Architecture")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CTI Corpus", "104 Reports")
    c2.metric("Segments", "5,991")
    c3, c4 = st.sidebar.columns(2)
    c3.metric("Graph Nodes", "1,400")
    c4.metric("Graph Edges", "329")
    st.sidebar.caption("NLI Verifier: `cross-encoder/nli-deberta-v3-base`")

    st.sidebar.markdown("---")
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

    # Main Area Header
    st.title("🛡️ CyberRAG: Live CTI Knowledge Graph & Verification Platform")
    st.markdown(
        "An evidence-grounded threat intelligence system that autonomously ingests new CTI reports, "
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

        # Left Column: Extracted Intelligence & Verification
        with col_left:
            st.subheader("📋 Threat Intelligence Analysis")

            # Overview Card
            st.markdown(
                f"""
                <div style="background-color: #161e2e; padding: 12px; border-radius: 8px; border: 1px solid #2d3748; margin-bottom: 12px;">
                    <b>Report:</b> {ext_res['title']}<br>
                    <b>Publisher:</b> <code>{ext_res['source']}</code> | <b>Publication Date:</b> <code>{ext_res['publication_date']}</code><br>
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
                    st.markdown(" ".join([f"<span class='badge-pill pill-actor'>{a.canonical_name}</span>" for a in actors]), unsafe_allow_html=True)
                else:
                    st.caption("No threat actor mentions identified.")

            with tab_malware:
                malware_tools = [e for e in ext_res["unique_entities"] if e.category in ("Malware", "Tool")]
                if malware_tools:
                    st.markdown(" ".join([
                        f"<span class='badge-pill pill-malware' title='{m.category}'>{m.canonical_name}</span>"
                        for m in malware_tools
                    ]), unsafe_allow_html=True)
                else:
                    st.caption("No malware or tool mentions identified.")

            with tab_targets:
                targets = [e for e in ext_res["unique_entities"] if e.category in ("Organization", "Location")]
                if targets:
                    st.markdown(" ".join([
                        f"<span class='badge-pill pill-org'>{t.canonical_name}</span>"
                        for t in targets
                    ]), unsafe_allow_html=True)
                else:
                    st.caption("No target organization or location mentions identified.")

            with tab_cves:
                cves = [e for e in ext_res["unique_entities"] if e.category in ("CVE", "Technique")]
                if cves:
                    st.markdown(" ".join([
                        f"<span class='badge-pill pill-cve'>{c.canonical_name}</span>"
                        for c in cves
                    ]), unsafe_allow_html=True)
                else:
                    st.caption("No CVEs or ATT&CK techniques identified.")

            st.markdown("---")

            # Evidence Verification Section
            st.markdown("##### 🔎 Evidence-Grounded Verification & Context Resolution")
            st.caption(
                "Cross-referencing candidate propositions against the 104-report corpus using Hybrid RAG, "
                "DeBERTa-v3 cross-encoder NLI, duplicate suppression, and CTI temporal contradiction resolution."
            )

            if not verifications:
                st.info("No semantic relationship triples were extracted from this report text.")
            else:
                seen_rendered_claims = set()
                for v in verifications:
                    canonical_key = v.get("claim", "").strip().lower()
                    if canonical_key in seen_rendered_claims:
                        continue
                    seen_rendered_claims.add(canonical_key)

                    status = v.get("status", "UNVERIFIED")
                    card_class = {
                        "SUPPORTED": "verif-card-supported",
                        "CONTRADICTED": "verif-card-contradicted",
                        "CONFLICTING_EVIDENCE": "verif-card-conflict",
                    }.get(status, "verif-card-insufficient")

                    status_icon = {
                        "SUPPORTED": "✅ SUPPORTED",
                        "CONTRADICTED": "❌ CONTRADICTED",
                        "CONFLICTING_EVIDENCE": "⚠️ CONFLICTING EVIDENCE",
                    }.get(status, "⚪ INSUFFICIENT EVIDENCE")

                    is_neg_assertion = v.get("polarity") == "negative"
                    pol_badge = " · <span style='color: #fdba74; font-weight: 600;'>⚠️ NEGATIVE ASSERTION (Provenance preserved; excluded from Knowledge Graph)</span>" if is_neg_assertion else ""

                    with st.container():
                        st.markdown(
                            f"""
                            <div class="{card_class}">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <h4 style="margin: 0; color: #ffffff;">{v['claim']}</h4>
                                    <span style="font-weight: bold; font-size: 14px;">{status_icon}</span>
                                </div>
                                <div style="font-size: 13px; color: #cbd5e1; margin-top: 6px;">
                                    <b>Extracted From:</b> <i>\"{v['extracted_from_sentence']}\"</i><br>
                                    <b>Trigger:</b> <code>{v['trigger']}</code> | <b>Extraction Confidence:</b> {v['confidence']}{pol_badge}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Tally row
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Supporting", f"{v['supporting_count']} ({v['non_duplicate_supporting_count']} unique)")
                        m2.metric("Contradicting", v['contradicting_count'])
                        m3.metric("Neutral", v['neutral_count'])
                        m4.metric("Duplicates Suppressed", v['duplicate_count'])

                        if v.get("unique_supporting_sources"):
                            num_pubs = len(v['unique_supporting_sources'])
                            st.caption(f"**Corroborating Reporting Sources:** {num_pubs} distinct publisher(s): {', '.join(v['unique_supporting_sources'])}")

                        # Context Resolution Breakdown
                        res = v.get("resolution", {})
                        if res and res.get("is_conflict"):
                            res_status = res.get("resolution_status", "CONFLICT")
                            delta_str = f" (+{res.get('temporal_delta_days')} days)" if res.get('temporal_delta_days') is not None else ""
                            supp_obs = f"{', '.join(res.get('supporting_sources', []))} ({', '.join(res.get('supporting_dates', []))})"
                            contra_obs = f"{', '.join(res.get('contradicting_sources', []))} ({', '.join(res.get('contradicting_dates', []))})"
                            st.warning(
                                f"**CTI Context Resolution Determination:** `{res_status}{delta_str}`\n\n"
                                f"• **Historical Supporting Observation:** {supp_obs}\n\n"
                                f"• **Later Contradicting Observation:** {contra_obs}\n\n"
                                f"• **Contextual Interpretation:** {res.get('explanation')}\n\n"
                                f"• **Recommended Action:** {res.get('recommended_action')}"
                            )
                        elif res and res.get("resolution_status") == "NO_CONFLICT" and status == "SUPPORTED":
                            num_pubs = len(v.get('unique_supporting_sources', []))
                            pub_str = f" from {num_pubs} distinct publisher(s)" if num_pubs else ""
                            st.success(f"Historical corpus consistently corroborates this intelligence{pub_str} without temporal conflict.")

                        # Expandable provenance inspector
                        with st.expander("🔍 Inspect Retrieved Evidence Passages & Provenance"):
                            ev_results = v.get("evidence_results", [])
                            if not ev_results:
                                st.write("No external passages retrieved.")
                            for idx, ev in enumerate(ev_results[:5], 1):
                                is_dup_str = f" ⚠️ [DUPLICATE of {ev.get('duplicate_of')}]" if ev.get("is_duplicate") else ""
                                st.markdown(
                                    f"**{idx}. [{ev.get('evidence_type', '').upper()}]{is_dup_str}** — "
                                    f"`{ev.get('source')}` ({ev.get('publication_date')})<br>"
                                    f"• Label: **{ev.get('label', '').upper()}** | Softmax Scores: "
                                    f"Entailment={ev.get('p_entailment')}, Contradiction={ev.get('p_contradiction')}, Neutral={ev.get('p_neutral')}<br>"
                                    f"• Passage: *\"{ev.get('evidence', '')[:200]}...\"*",
                                    unsafe_allow_html=True,
                                )

            # Interactive Custom Hypothesis Testing
            st.markdown("---")
            st.markdown("##### 🧪 Test Custom / Opposing Proposition Against Historical Corpus")
            st.caption("Directly evaluate any proposition against the 104-report corpus to demonstrate NLI contradiction detection or temporal supersedence.")

            default_hypo = "Emotet USES Trickbot." if "Emotet" in ext_res["title"] else "APT28 never targeted NATO."
            custom_claim = st.text_input("Enter proposition to evaluate:", value=default_hypo)
            test_hypo_btn = st.button("Evaluate Hypothesis Against Corpus", key="eval_custom_hypo")

            if test_hypo_btn and custom_claim:
                with st.spinner("Retrieving evidence & running NLI cross-encoder verification..."):
                    hypo_rag = pipeline.ask(custom_claim, verify=True, verification_claims=[custom_claim])
                    verif_data = hypo_rag.get("verification", {}).get("claims", [])
                    if verif_data:
                        hv = verif_data[0]
                        h_status = hv.get("status", "UNVERIFIED")
                        h_card_class = {
                            "SUPPORTED": "verif-card-supported",
                            "CONTRADICTED": "verif-card-contradicted",
                            "CONFLICTING_EVIDENCE": "verif-card-conflict",
                        }.get(h_status, "verif-card-insufficient")
                        h_icon = {
                            "SUPPORTED": "✅ SUPPORTED",
                            "CONTRADICTED": "❌ CONTRADICTED",
                            "CONFLICTING_EVIDENCE": "⚠️ CONFLICTING EVIDENCE",
                        }.get(h_status, "⚪ INSUFFICIENT EVIDENCE")

                        st.markdown(
                            f"""
                            <div class="{h_card_class}">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <h4 style="margin: 0; color: #ffffff;">{hv['claim']}</h4>
                                    <span style="font-weight: bold; font-size: 14px;">{h_icon}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        hm1, hm2, hm3, hm4 = st.columns(4)
                        hm1.metric("Supporting", hv.get("supporting_count", 0))
                        hm2.metric("Contradicting", hv.get("contradicting_count", 0))
                        hm3.metric("Neutral", hv.get("neutral_count", 0))
                        hm4.metric("Duplicates Suppressed", hv.get("duplicate_count", 0))

                        h_res = hv.get("resolution", {})
                        if h_res and h_res.get("is_conflict"):
                            st.warning(
                                f"**CTI Context Resolution:** `{h_res.get('resolution_status')}`\n\n"
                                f"• {h_res.get('explanation')}\n\n"
                                f"• Recommended Action: {h_res.get('recommended_action')}"
                            )
                        elif h_status == "CONTRADICTED":
                            contra_publishers = sorted(list({e.get("source") for e in hv.get("evidence_results", []) if e.get("label") == "contradiction" and e.get("source")}))
                            pub_str = f" from {len(contra_publishers)} distinct publisher(s): {', '.join(contra_publishers)}" if contra_publishers else ""
                            st.error(f"Textual evidence in the historical corpus directly refutes this hypothesis ({hv.get('contradicting_count', 0)} contradicting passages{pub_str}).")
                        elif h_status == "SUPPORTED":
                            supp_publishers = sorted(list({e.get("source") for e in hv.get("evidence_results", []) if e.get("label") == "entailment" and e.get("source")}))
                            supp_pub_str = f" from {len(supp_publishers)} distinct publisher(s): {', '.join(supp_publishers)}" if supp_publishers else ""
                            st.success(f"Historical corpus consistently corroborates this intelligence ({hv.get('supporting_count', 0)} supporting passages{supp_pub_str}) without temporal conflict.")

        # Right Column: Live Knowledge Graph
        with col_right:
            st.subheader("🕸️ Live CTI Knowledge Graph")

            # Metrics
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Total Nodes", merged["total_nodes"])
            g2.metric("New Nodes Added", f"+{merged['new_nodes_count']}")
            g3.metric("Total Edges", merged["total_edges"])
            g4.metric("New Edges Added", f"+{merged['new_edges_count']}")

            # Legend
            st.markdown(
                """
                <div style="font-size: 12px; margin-bottom: 8px;">
                    <span style="color: #00f5d4; font-weight: bold;">✦ Neon Cyan: Newly Ingested</span> | 
                    <span style="color: #e63946; font-weight: bold;">● Red: Threat Actor</span> | 
                    <span style="color: #7209b7; font-weight: bold;">● Purple: Malware</span> | 
                    <span style="color: #f77f00; font-weight: bold;">● Orange: Tool</span> | 
                    <span style="color: #fcbf49; font-weight: bold;">● Yellow: CVE</span> | 
                    <span style="color: #2a9d8f; font-weight: bold;">● Green: Target Org</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            focus_ids = [e.entity_id for e in ext_res["unique_entities"]]
            subgraph_html = processor.generate_subgraph_html(
                merged["graph"],
                focus_entity_ids=focus_ids,
                max_hops=ego_depth,
                height="750px",
            )
            components.html(subgraph_html, height=780, scrolling=False)

            with st.expander("ℹ️ Graph Inspection & Ingestion Audit"):
                st.write(f"**Focused Report Entity Seeds ({len(focus_ids)}):**")
                st.write(", ".join(focus_ids))
                if merged.get("new_edge_keys"):
                    st.write("**Newly Established Edges:**")
                    for ek in merged["new_edge_keys"]:
                        st.markdown(f"- `✦ {ek}`")


if __name__ == "__main__":
    main()
