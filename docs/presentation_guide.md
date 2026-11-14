# CyberRAG: Final Committee Presentation & Live Demo Walkthrough

**15-Minute Presentation Guide & Defense Strategy**  
*Integrated System of Person 1 (Knowledge Engineering), Person 2 (Hybrid & Graph Retrieval), and Person 3 (Verification & Context Resolution)*

---

## 1. Committee Presentation Structure (15 Minutes)

| Time | Slide / Topic | Key Message to Deliver |
| :--- | :--- | :--- |
| **00:00 – 02:30** | **The Problem with LLMs in CTI** | Generic RAG suffers from hallucinations, ungrounded graph triples, and false consensus when reports duplicate sentences. CTI requires strict provenance and temporal contradiction awareness. |
| **02:30 – 05:00** | **System Architecture & Division of Labor** | Person 1 (NER + Relation Extractor v2 + Knowledge Graph) $\rightarrow$ Person 2 (Hybrid RAG + Graph Traversal + Fusion) $\rightarrow$ Person 3 (Cross-Encoder NLI + Provenance Adapter + Deduplication + Context Resolution). |
| **05:00 – 10:00** | **Live Interactive Platform Demo** | Run `app.py` through the 4 live demo scenarios (Upload $\rightarrow$ Extract $\rightarrow$ Expand Graph $\rightarrow$ Verify). |
| **10:00 – 12:30** | **Core Research Contributions** | 1) Clause-aware extraction with vendor suppression; 2) Graph/text duplicate suppression; 3) 4-tier CTI contradiction taxonomy (`TEMPORAL_SUPERSEDENCE`, `CAMPAIGN_DIVERGENCE`, `GENUINE_ATTRIBUTION_DISPUTE`). |
| **12:30 – 15:00** | **Evaluation Metrics & Q&A** | 8,368 entities, 374 assertions, 1,400 graph nodes, 329 edges, 54/54 passing unit tests, 0.99+ NLI precision on CTI benchmarks. |

---

## 2. Live Demo Script (Step-by-Step in `app.py`)

Launch command:
```powershell
streamlit run app.py
```

### Scenario 1: APT28 Campaign Targeting NATO (Corroborated Intelligence)
1. **Select**: `Preset 1: APT28 Campaign Targeting NATO (Corroborated)`.
2. **Click**: `🚀 Ingest, Update Graph & Verify`.
3. **What to highlight on screen**:
   - **Left Panel (Cyber NLP)**:
     - 4 unique entities extracted (`ThreatActor::APT28`, `Malware::Zebrocy`, `Organization::NATO`, `Location::Europe`).
     - Exactly 2 high-precision relations extracted: `APT28 USES Zebrocy` (trigger: *deploying*) and `APT28 TARGETS NATO` (trigger: *targeted*).
     - **Point out to committee**: Spurious spatial targeting (`Zebrocy TARGETS Europe`) was suppressed because *across* is a spatial preposition, not an intentional targeting verb.
   - **Verification Cards**:
     - `APT28 USES Zebrocy` $\rightarrow$ **`SUPPORTED`** (7 supporting passages).
     - `APT28 TARGETS NATO` $\rightarrow$ **`SUPPORTED`** (4 supporting, 4 neutral). Show how NLI correctly identifies using NATO training as a phishing decoy as `NEUTRAL`, while actual victim targeting is `ENTAILMENT`.
   - **Right Panel (Knowledge Graph)**:
     - Shows existing 1,400-node graph connected to `APT28`, `Zebrocy`, and `NATO`.

### Scenario 2: Novel Threat Actor & Malware (Dynamic Graph Expansion)
1. **Select**: `Preset 4: Novel Threat Intelligence (Sandworm & Cyclops Blink - Dynamic Graph Expansion)`.
2. **Click**: `🚀 Ingest, Update Graph & Verify`.
3. **What to highlight on screen**:
   - **Right Panel (Graph Expansion)**:
     - Metrics display: **`Total Nodes: 1,402 (+2 New Nodes)`**, **`Total Edges: 334 (+5 New Edges)`**.
     - Point to the **✦ Glowing Neon Cyan (`#00f5d4`) Nodes** for `Sandworm` and `Cyclops Blink`.
     - Explain that the underlying baseline graph on disk (`data/cti_knowledge_graph_trusted.json`) remains completely unmutated, proving safe non-destructive live intelligence ingestion.

### Scenario 3: Law Enforcement Takedown (Temporal Contradiction & Resolution)
1. **Select**: `Preset 2: Emotet Takedown`.
2. **Click**: `🚀 Ingest, Update Graph & Verify`.
3. **Scroll down to**: `🧪 Test Custom / Opposing Proposition Against Historical Corpus`.
4. **Enter**: `"Emotet never downloaded or executed Trickbot."` $\rightarrow$ Click `Evaluate Hypothesis Against Corpus`.
5. **What to highlight on screen**:
   - Status: **`❌ CONTRADICTED`** (9 contradicting passages from Intel 471 and Zscaler).
   - **CTI Context Resolution Box**:
     - Status: **`TEMPORAL_SUPERSEDENCE`**
     - Time Delta: **220 days**.
     - **Explain to committee**: *"The model refutes the statement based on 2020 intelligence, but our Context Resolver notices the 2021 report post-dates the 2020 activity by 220 days. This is an operational disruption / infrastructure takedown, NOT a historical lie."*

### Scenario 4: TrickBot-Emotet Partnership (Duplicate Evidence Suppression)
1. **Select**: `Preset 3: TrickBot-Emotet Malware Delivery Partnership`.
2. **Click**: `🚀 Ingest, Update Graph & Verify`.
3. **What to highlight on screen**:
   - Status: **`✅ SUPPORTED`**.
   - Show metric: **`Duplicates Suppressed: 3`**.
   - Explain to committee: *"A graph assertion and a document passage frequently contain the same sentence. Rather than counting them as two independent corroborating sources, our pipeline flags duplicates and reports only genuine multi-source consensus (Intel 471 & Zscaler)."*

---

## 3. Anticipated Committee Questions & Defenses

### Q1: "Why use a DeBERTa cross-encoder for NLI instead of prompting a large LLM like GPT-4?"
> **Answer**:  
> *"Generative LLMs are prone to sycophancy, position bias, and non-deterministic text generation. A dedicated NLI cross-encoder (`nli-deberta-v3-base`) models the joint attention between premise and hypothesis simultaneously across all transformer layers. It outputs reproducible softmax distributions in sub-second inference time without API costs, data leakage, or hallucination."*

### Q2: "Why can't you just average NLI probabilities into a factual confidence score?"
> **Answer**:  
> *"Neural softmax scores represent model token classification confidence regarding whether text A entails text B. They do NOT represent empirical historical truth. In intelligence analysis, a vendor can write a sentence with 100% linguistic clarity about a false rumor. Conflating neural confidence with ground truth violates scientific integrity. We report source provenance, corroborating vendor count, and raw softmax scores separately."*

### Q3: "How do you distinguish a genuine CTI contradiction from a temporal update?"
> **Answer**:  
> *"NLI models operate on static linguistic semantics—they cannot understand that a threat group changed tools between 2020 and 2021. Our Context Resolver layer inspects publication dates and campaign scope across conflicting records. When reports are separated by $>90$ days, it resolves the conflict as `TEMPORAL_SUPERSEDENCE` rather than flagging an erroneous attribution fight."*

### Q4: "How does the system prevent attributing malicious attacks to cybersecurity vendors?"
> **Answer**:  
> *"Reporting vendors (e.g., Mandiant, CrowdStrike, Zscaler) frequently appear in sentences like 'Discovered by Zscaler researchers.' Our relation extractor implements category-scoped vendor suppression: whenever an Organization entity appears in a passive observation context, it is suppressed from the subject or object of a `TARGETS` or `USES` relationship."*
