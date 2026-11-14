# CyberRAG: Committee Presentation & Live Defense Guide

**15-Minute Defense Strategy & Live Demo Sequence**  
*Integrated Platform: Hybrid Graph RAG, Cross-Encoder NLI Verification, Provenance Preservation, and Context-Aware Contradiction Resolution*

---

## 1. Committee Presentation Structure (15 Minutes)

| Time | Slide / Topic | Key Message to Deliver |
| :--- | :--- | :--- |
| **00:00 – 02:30** | **The Problem with LLMs in CTI** | Generic LLMs and naive RAG pipelines suffer from hallucinations, ungrounded graph triples, false consensus from duplicated vendor reports, and temporal naivety when operational realities change. |
| **02:30 – 05:00** | **System Architecture** | Multi-stage pipeline: Cyber NLP (NER + clause-aware relation extraction) $\rightarrow$ Typed Knowledge Graph $\rightarrow$ Hybrid Dense/BM25/Graph Retrieval $\rightarrow$ DeBERTa-v3 Cross-Encoder NLI $\rightarrow$ Multi-publisher Claim Aggregator $\rightarrow$ Context Resolver. |
| **05:00 – 10:30** | **Live Interactive Platform Demo** | Run `app.py` through the 4 live demo scenarios (Corroboration $\rightarrow$ Temporal Supersedence $\rightarrow$ Duplicate Suppression $\rightarrow$ Dynamic Graph Expansion). |
| **10:30 – 12:30** | **Core Research Contributions** | 1) Clause-aware extraction with vendor & spatial trigger suppression; 2) Intra-publisher duplicate passage detection; 3) 4-tier CTI contradiction taxonomy (`TEMPORAL_SUPERSEDENCE`, `CAMPAIGN_DIVERGENCE`, `GENUINE_ATTRIBUTION_DISPUTE`); 4) Structured provenance-grounded evidence reporting. |
| **12:30 – 15:00** | **Evaluation Metrics & Q&A** | 1,400 graph nodes, 329 edges, 374 validated assertions, 5,991 corpus segments, 60/60 passing unit tests, and real-world AnnoCTR empirical benchmark evaluation. |

---

## 2. Live Demo Sequence (Step-by-Step in `app.py`)

Launch command:
```bash
streamlit run app.py
```

### Scenario 1: APT28 Campaign Targeting NATO (Corroborated Intelligence)
1. **Select**: `Preset 1: APT28 Campaign Targeting NATO (Corroborated)`.
2. **Click**: `🚀 Ingest, Update Graph & Verify`.
3. **What to highlight on screen**:
   * **Left Panel (Cyber NLP)**:
     - 4 unique entities extracted (`ThreatActor::APT28`, `Malware::Zebrocy`, `Organization::NATO`, `Location::Europe`).
     - Exactly 2 high-precision relations extracted: `APT28 USES Zebrocy` (trigger: *deploying*) and `APT28 TARGETS NATO` (trigger: *targeted*).
     - **Point out to committee**: Spurious spatial targeting (`Zebrocy TARGETS Europe`) was suppressed because *across* is a spatial preposition, not an intentional targeting predicate.
   * **Verification Cards**:
     - `APT28 USES Zebrocy` $\rightarrow$ **`SUPPORTED`** (7 supporting passages).
     - `APT28 TARGETS NATO` $\rightarrow$ **`SUPPORTED`** (4 supporting, 4 neutral). Show how NLI correctly identifies using NATO training as a phishing decoy as `NEUTRAL`, while actual victim targeting is `ENTAILMENT`.
   * **Right Panel (Knowledge Graph)**:
     - Shows the connected subgraph around `APT28`, `Zebrocy`, and `NATO` within the 1,400-node graph.

### Scenario 2: Law Enforcement Takedown (Temporal Contradiction & Supersedence)
1. **Select**: `Preset 2: Emotet Infrastructure Takedown (Europol / CISA 2021)`.
2. **Click**: `🚀 Ingest, Update Graph & Verify`.
3. **What to highlight on screen**:
   * **Automated Hypothesis Evaluation**:
     - Proposition evaluated: `"Emotet USES TrickBot."`
     - **Historical Baseline Intelligence (2020-10-15, Intel 471)**: Supported with `ENTAILMENT` (Emotet distributed TrickBot).
     - **Newly Ingested Post-Seizure Intelligence (2021-05-20, Europol / CISA)**: Contradicted with `CONTRADICTION` (score: `0.9951`, infrastructure seized, zero activity).
   * **Overall Claim Status**: **`⚠️ CONFLICTING_EVIDENCE`**.
   * **Context Resolver Output**:
     - Classification: **`TEMPORAL_SUPERSEDENCE (+217 days, CONTRA_LATER)`**.
     - Time Delta: **217 days** (October 15, 2020 $\rightarrow$ May 20, 2021).
     - **Explain to committee**: *"Standard NLI would treat this as a factual error or lie. CyberRAG's Context Resolver compares publication dates and resolves it as an operational shift over time: the earlier relationship was historically accurate in 2020, but superseded by law enforcement disruption in 2021."*
   * **Negative Assertion Retention**:
     - Show that the negative assertion (`Emotet never downloaded or executed TrickBot post-takedown`) retains full provenance without corrupting the positive baseline graph.

### Scenario 3: TrickBot-Emotet Partnership (Duplicate Evidence Suppression)
1. **Select**: `Preset 3: TrickBot-Emotet Malware Delivery Partnership`.
2. **Click**: `🚀 Ingest, Update Graph & Verify`.
3. **What to highlight on screen**:
   * Status: **`✅ SUPPORTED`**.
   * Show metric: **`Duplicates Suppressed: 3`**.
   * Multi-Publisher Consensus: Reports consensus across **2 distinct publishers** (*Intel 471* and *Zscaler*).
   * Explain to committee: *"A graph assertion and a document passage frequently share verbatim phrasing. Rather than treating identical text as independent corroboration, our pipeline suppresses duplicate passages and tallies genuine multi-publisher consensus."*

### Scenario 4: Sandworm & Cyclops Blink (Novel Threat Intel & Dynamic Graph Expansion)
1. **Select**: `Preset 4: Novel Threat Intelligence (Sandworm & Cyclops Blink - Dynamic Graph Expansion)`.
2. **Click**: `🚀 Ingest, Update Graph & Verify`.
3. **What to highlight on screen**:
   * **Verification Cards (False Contradiction Prevention)**:
     - All 3 newly extracted claims (`Sandworm USES Cyclops Blink`, `Sandworm TARGETS Energy Sector`, `Sandworm TARGETS Ukraine`) evaluate to **`⚪ NO EVIDENCE IN CORPUS`** (`INSUFFICIENT_EVIDENCE`).
     - **Explain to committee**: *"Because Sandworm and Cyclops Blink are novel entities not present in the historical 104-report corpus, our entity-aware retrieval filter prevents unrelated corpus passages (e.g., Taurus sandbox evasion) from reaching the NLI model and falsely triggering high-confidence contradictions. The system correctly acknowledges zero prior baseline evidence rather than fabricating a conflict."*
   * **Right Panel (Graph Expansion)**:
     - Metrics display: **`Total Nodes: 1,402 (+2 New Nodes)`**, **`Total Edges: 334 (+5 New Edges)`**.
     - Point to the **✦ Glowing Neon Cyan (`#00f5d4`) Diamond Nodes** for `Sandworm` and `Cyclops Blink`.
     - Explain that the underlying baseline graph on disk (`data/cti_knowledge_graph_trusted.json`) remains completely untouched, proving safe in-memory session expansion.

### Scenario 5: Arbitrary PDF / TXT Upload (Generalization)
1. Upload a custom plain-text or PDF report (e.g., `custom_analyst_report.txt`).
2. Show that the entire pipeline—text extraction, indicator refanging, NER, relation extraction, graph expansion, and NLI verification—executes end-to-end without crashing.

---

## 3. Anticipated Committee Questions & Defenses

### Q1: "Why use a DeBERTa cross-encoder for NLI instead of prompting GPT-4?"
> **Answer**:  
> *"Generative LLMs are prone to sycophancy, position bias, and non-deterministic text generation. A dedicated NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`) models joint cross-attention between premise and hypothesis simultaneously across all transformer layers. It outputs reproducible softmax distributions in sub-second inference time with zero API costs, zero external data leakage, and zero hallucination."*

### Q2: "Why can't you just average NLI probabilities into a factual truth score?"
> **Answer**:  
> *"Softmax scores represent linguistic textual alignment (whether text A logically entails text B), not empirical historical truth. In cyber threat intelligence, a vendor can write an erroneous rumor with 100% syntactic confidence. Conflating neural confidence with ground truth violates scientific integrity. We report source provenance, distinct publisher counts, and raw softmax scores separately."*

### Q3: "Why doesn't CyberRAG generate natural-language answers with an LLM?"
> **Answer**:  
> *"Our AnswerGenerator is deliberately designed to produce structured, provenance-grounded intelligence syntheses. In mission-critical CTI analysis, an analyst needs traceable citations: exact Report IDs, publishers, publication dates, and verbatim evidence sentences. Unconstrained generative text introduces subtle hallucinations that can mislead defenders. We prioritize auditability over generative fluency."*

### Q4: "What does your Graph Retrieval actually do?"
> **Answer**:  
> *"Graph retrieval in CyberRAG is implemented as entity-linked assertion lookup and multi-hop neighborhood traversal. It identifies query entities, traverses adjacent validated graph edges, and retrieves the underlying verbatim evidence sentences. We do not claim unconstrained semantic graph reasoning; every graph triple is strictly grounded in an ingested CTI source sentence."*

### Q5: "How do you distinguish a genuine CTI contradiction from a temporal update?"
> **Answer**:  
> *"NLI models operate on static linguistic semantics—they cannot understand that threat actors adapt over time. When conflicting intelligence is detected, our Context Resolver inspects publication dates and campaign scope. When conflicting records are separated by $>90$ days, the system classifies the conflict as `TEMPORAL_SUPERSEDENCE` rather than flagging an erroneous attribution dispute."*

### Q6: "How do you prevent reporting cybersecurity vendors from being flagged as threat actors or targets?"
> **Answer**:  
> *"Reporting vendors (e.g., Mandiant, CrowdStrike, Zscaler) frequently appear in phrases like 'observed by Mandiant researchers.' Our relation extractor implements category-scoped vendor suppression: whenever an Organization entity appears in a passive observation context, it is suppressed from becoming the subject or object of a `TARGETS` or `USES` relationship."*

---

## 4. Evaluation Summary for Committee

When presenting evaluation results, distinguish between evaluation settings:

1. **Synthetic Controlled Challenge Benchmark**:
   * Evaluated under GroupKFold (Leave-One-Report-Out) cross-validation: **F1 = 0.9809**.
   * Demonstrates precision on canonical entities with dense relational triggers.
2. **Official Bosch Research AnnoCTR Benchmark (LREC-COLING 2024)**:
   * Evaluated on official train (7,570 sentences) and test (3,059 sentences) splits with **strictly 0% document overlap and zero leakage**: **F1 = 0.4396**.
   * Represents real-world noisy CTI performance, delivering a **+28.5% relative F1 improvement** over dictionary gazetteers (0.3421).
3. **Unseen CTI Generalization**:
   * Evaluated on 4 novel threat reports: **F1 = 0.9200** with **100% provenance metadata compliance**.
4. **Unit Test Suite**:
   * **60/60 unit tests passing** across 13 test suites (`python run_tests.py`).
