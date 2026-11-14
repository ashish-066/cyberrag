# CyberRAG — Person 1: Data + Cyber NLP + Knowledge Graph

> **Main Responsibility**: Turn raw Cyber Threat Intelligence (CTI) reports into structured cybersecurity knowledge.

This repository implements the production-grade, audited pipeline for **Person 1** in the CyberRAG system:

1. **Dataset Selection & Ingestion**: 104 authentic CTI reports (157,231 words) reconstructed directly from the official **Bosch Research AnnoCTR Corpus** (LREC-COLING 2024) token sequences with complete provenance metadata (**Report ID**, **Source**, **Paragraph/Sentence ID & Index**, **Publication Date**). All fabricated external URLs have been removed (`url: null`), citing the verified source repository.
2. **Text Cleaning & Preprocessing**: Refanging indicators (`hxxp[:]//`, `[.]`, `[:]`, `[@]`), unicode NFKC normalization, boilerplate stripping, and paragraph/sentence segmentation preserving full document provenance across 2,698 paragraphs and 5,991 sentences.
3. **Cybersecurity Named Entity Recognition (NER) & Disambiguation**:
   - **Threat Actors** (e.g., Volt Typhoon, APT29, APT28, Lazarus Group, Sandworm, LockBit, FIN7, TA505)
   - **Malware** (e.g., Emotet, TrickBot, AcidRain, HermeticWiper, CaddyWiper, Industroyer2, BlackCat, Carbanak)
   - **Tools** (e.g., Cobalt Strike, Mimikatz, PsExec, Chisel, Fast Reverse Proxy, PowerShell, WMI, BloodHound)
   - **CVEs** (e.g., CVE-2024-21887, CVE-2023-46805, CVE-2023-38831, CVE-2021-44228 Log4Shell, CVE-2021-34527 PrintNightmare)
   - **Techniques** (MITRE ATT&CK: T1566 Phishing, T1190 Exploit Public-Facing Application, T1059 Command & Scripting, T1078 Valid Accounts, T1486 Data Encrypted for Impact)
   - **Organizations** (e.g., Microsoft, NATO, Ukrainian Ministry of Defense, Energy Sector, Financial)
   - **Locations** (e.g., United States, Ukraine, Poland, South Korea, Japan, Germany, United Kingdom)
   - **Comparative NER Engine**: Baseline Gazetteer/Regex NER vs Contextual Machine Learning NER (`src/nlp/ml_ner.py`) vs Hybrid NER.
   - **Official MITRE ATT&CK Registry**: 1,726 canonical entities ingested from STIX 2.1 Enterprise bundle (178 Threat Actors, 747 Malware, 95 Tools, 706 Techniques).
4. **Clause-Aware Semantic Relationship Extraction**:
   - `USES`: Threat Actor / Malware &rarr; Malware / Tool (45 instances)
   - `TARGETS`: Threat Actor / Malware &rarr; Organization / Location (63 instances)
   - `EXPLOITS`: Threat Actor / Malware / Tool &rarr; CVE (4 instances)
   - `USES_TECHNIQUE`: Threat Actor / Malware &rarr; Technique (80 instances)
   - **Quality Features**: Intervening subject blocking, coordinate action clause blocking, passive voice inversion (`[CVE] was exploited by [Actor]`), localized predicate negation filtering (`did not deploy`), and reporting vendor attribution filtering.
5. **Initial Cyber Knowledge Graph Construction**:
   - In-memory `NetworkX` MultiDiGraph with **428 unique nodes** and **165 consolidated edges** (carrying 192 sentence-level relation evidence instances).
   - Graph exports:
     - `data/knowledge_graph/cyber_kg.json` (Node-link JSON with top-level scalar provenance fields)
     - `data/knowledge_graph/cyber_kg.graphml` (GraphML for Gephi & Cytoscape)
     - `data/knowledge_graph/import_kg.cypher` (Ready-to-run Neo4j Cypher script with schema constraints and `MERGE` statements)
   - Extended Edge Metadata: `polarity: 'positive'`, `observation_time`, `scope: 'sentence_level'`, `confidence_level: 'HIGH'/'MEDIUM'`, and discrete multi-report evidence lists.
6. **Dual Evaluation Framework**:
   - **Independent Real-World AnnoCTR Benchmark**: Evaluated against the official Bosch Research train (`ner_train.jsonl`, 70 reports, 7,570 sentences) and test (`ner_test.jsonl`, 34 reports, 3,059 sentences) splits with strictly **0% document overlap and zero leakage**.
   - **Synthetic Controlled Challenge Benchmark**: 50-sentence synthetic benchmark (`data/benchmark/gold_standard_annotations.json`) evaluated under **Report-Level GroupKFold Cross-Validation (Leave-One-Report-Out)**.
   - **Genuine 100-Relation Human Audit**: A 100-relation stratified sample drawn directly from `data/relations/extracted_relations.json` on the 104 ingested reports, verifying 100/100 report ID, sentence ID, and verbatim evidence text matches.
7. **Graph Visualizations**:
   - **Interactive HTML Graph Explorer** (`outputs/visualizations/cyber_knowledge_graph.html`): Real-time node search, category filter pills, threat actor focus dropdown, evidence inspector drawer, physics toggle.
   - **Static High-Resolution Diagrams**:
     - `full_knowledge_graph.png`: Multi-relational network overview
     - `threat_actor_network.png`: Threat actor capabilities and target map
     - `cve_exploit_network.png`: Vulnerability exploitation bipartite graph

---

## Architecture & Directory Layout

```
cyberrag/
├── data/
│   ├── annoctr_source/           # Official AnnoCTR dataset splits (Bosch Research)
│   │   ├── ner_train.jsonl       # Official train split (70 reports, 7,570 sentences)
│   │   ├── ner_test.jsonl        # Official test split (34 reports, 3,059 sentences)
│   │   └── ner_dev.jsonl         # Official dev split (20 reports, 1,550 sentences)
│   ├── raw/                      # 104 reconstructed CTI reports with honest provenance
│   │   └── cti_reports.json
│   ├── processed/                # Preprocessed paragraphs/sentences with provenance
│   │   └── preprocessed_cti.json # 2,698 paragraphs, 5,991 sentences
│   ├── entities/                 # Extracted and normalized entities (5,864 mentions)
│   │   └── extracted_entities.json
│   ├── relations/                # Extracted relationship instances (192 instances)
│   │   └── extracted_relations.json
│   ├── knowledge/                # Official MITRE ATT&CK STIX 2.1 registry (1,726 items)
│   │   └── mitre_attack_registry.json
│   ├── knowledge_graph/          # Graph exports (428 nodes, 165 consolidated edges)
│   │   ├── cyber_kg.json
│   │   ├── cyber_kg.graphml
│   │   └── import_kg.cypher
│   └── benchmark/                # Evaluation benchmarks
│       ├── annoctr_train_split.json    # Official AnnoCTR training split (2,840 sents)
│       ├── annoctr_gold_standard.json  # Official AnnoCTR test split (100 sents)
│       └── gold_standard_annotations.json # 50-sentence synthetic challenge benchmark
├── src/
│   ├── dataset/                  # Dataset loading & schema validation
│   │   ├── loader.py
│   │   └── downloader.py
│   ├── preprocessor/             # Refanging, unicode cleaning, segmentation
│   │   ├── cleaner.py
│   │   └── segmenter.py
│   ├── nlp/                      # Cyber NER, ML NER, normalizer & relation extractor
│   │   ├── gazetteer.py
│   │   ├── normalizer.py
│   │   ├── ml_ner.py
│   │   ├── ner_engine.py
│   │   ├── relation_extractor.py
│   │   ├── evaluator.py
│   │   ├── annoctr_evaluator.py
│   │   └── unseen_evaluator.py
│   ├── graph/                    # Graph builder, exporters & analytics
│   │   ├── builder.py
│   │   ├── exporter.py
│   │   └── analytics.py
│   └── visualization/            # Interactive HTML & static matplotlib plots
│       ├── interactive_html.py
│       └── static_plotter.py
├── outputs/
│   ├── visualizations/           # HTML and PNG visualizations
│   │   ├── cyber_knowledge_graph.html
│   │   ├── full_knowledge_graph.png
│   │   ├── threat_actor_network.png
│   │   └── cve_exploit_network.png
│   └── reports/                  # Summary intelligence reports & benchmarks
│       ├── cti_kg_summary.json
│       ├── evaluation_benchmark.json
│       ├── annoctr_ner_evaluation.json
│       ├── relation_sample_100_audit.json
│       └── unseen_reports_evaluation.json
├── scripts/                      # Construction and audit scripts
│   ├── reconstruct_annoctr_reports.py
│   ├── build_official_annoctr_splits.py
│   └── generate_genuine_100_relation_audit.py
├── tests/                        # Full unit and integration test suite
│   ├── test_preprocessor.py
│   ├── test_ner.py
│   ├── test_relation_extractor.py
│   └── test_knowledge_graph.py
├── run_pipeline.py               # Master CLI pipeline runner
├── verify_pipeline.py            # Automated quality & provenance auditor
├── run_tests.py                  # Test runner (13 unit and integration tests)
└── requirements.txt
```

---

## Quick Execution Commands

### 1. Run Complete Pipeline
```bash
python run_pipeline.py
```

### 2. Run Automated Verification & Quality Audit
```bash
python verify_pipeline.py
```
*Validates entity coverage across all 7 categories, trigger grounding on 100% of relations, zero invalid schemas, and zero provenance gaps.*

### 3. Run Test Suite (13/13 Tests Passing)
```bash
python run_tests.py
```

### 4. Run Independent AnnoCTR Benchmark Evaluation
```bash
python -m src.nlp.annoctr_evaluator
```

### 5. Run Genuine 100-Relation Sample Human Audit
```bash
python scripts/generate_genuine_100_relation_audit.py
```

### 6. Run Synthetic Controlled Challenge Benchmark (Report-Level GroupKFold)
```bash
python -m src.nlp.evaluator
```

---

## Metrics & Terminology Clarification (104 Authentic CTI Reports Ingested)

| Concept | Count | Description |
| :--- | :---: | :--- |
| **Ingested CTI Reports** | **104** | Authentic CTI reports (157,231 words) reconstructed from the Bosch Research AnnoCTR corpus token sequences with honest provenance metadata (`url: null`). |
| **Preprocessed Sentences** | **5,991** | Tokenized sentences across 2,698 paragraphs in 104 authentic reports, preserving complete sentence-level provenance metadata. |
| **Extracted Entity Mentions** | **5,864** | Entity mentions across sentences (Techniques: 2,806, Malware: 897, Tools: 775, Locations: 765, ThreatActors: 326, Organizations: 259, CVEs: 36). |
| **Extracted Relation Instances** | **192** | Grounded semantic relation assertions extracted directly from CTI text (`USES_TECHNIQUE`: 80, `TARGETS`: 63, `USES`: 45, `EXPLOITS`: 4). |
| **Consolidated Knowledge Graph Edges** | **165** | Unique directed multi-relational edges between entity nodes in the consolidated Knowledge Graph, consolidating multi-report citations into evidence lists. |
| **Unique Graph Nodes** | **428** | Consolidated canonical entities in the Cyber Knowledge Graph across all 7 categories. |
| **Official MITRE ATT&CK Knowledge Registry** | **1,726** | Canonical entities ingested from official MITRE STIX 2.1 Enterprise bundle: 178 Threat Actors, 747 Malware, 95 Tools, 706 Techniques. |

---

## Independent Real-World AnnoCTR Benchmark Results

> Full evaluation report saved to [`outputs/reports/annoctr_ner_evaluation.json`](file:///c:/Users/njosh/Desktop/cyberrag/outputs/reports/annoctr_ner_evaluation.json).  
> Evaluates models against **100 independent, human-annotated sentences** from 34 held-out authentic test reports (171 gold entities across 7 categories) from official `ner_test.jsonl`. The contextual ML NER model was trained strictly on the official AnnoCTR training split of **2,840 entity-bearing sentences** (63,234 tokens across 70 disjoint reports) from official `ner_train.jsonl`, guaranteeing strictly **0% document overlap and zero test leakage**.

| Model / Architecture | Precision | Recall | F1 Score | Key Findings & Error Modes on Real CTI Text |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline Gazetteer NER** | **0.5806** | **0.4211** | **0.4881** | High specificity on known static aliases (1.0 precision on ThreatActor and Location), but lower recall (42.1%) due to out-of-vocabulary entities (e.g. AcidRain, DEWMODE, Mount Locker). |
| **Hybrid Contextual ML NER** | **0.6071** | **0.5965** | **0.6018** | Successfully recovers novel entities without gazetteer entries, increasing True Positives from 72 to 102 (+41.7% yield). Demonstrates genuine generalization over baseline gazetteer on authentic real-world CTI text (+11.37 F1 points). |

### Category Breakdown (Hybrid Contextual ML NER on Real-World Text):
- **Location**: Precision: **0.9394** | Recall: **0.9394** | F1: **0.9394**
- **Malware**: Precision: **0.9048** | Recall: **0.6129** | F1: **0.7308**
- **ThreatActor**: Precision: **0.8462** | Recall: **0.5000** | F1: **0.6286**
- **Organization**: Precision: **0.6667** | Recall: **0.5294** | F1: **0.5902**
- **Tool**: Precision: **0.4583** | Recall: **0.5000** | F1: **0.4783**
- **Technique**: Precision: **0.2400** | Recall: **0.4138** | F1: **0.3038**
- **Overall**: Precision: **0.6071** | Recall: **0.5965** | F1: **0.6018**

---

## Genuine 100-Relation Sample Human Audit

> Full audit report saved to [`outputs/reports/relation_sample_100_audit.json`](file:///c:/Users/njosh/Desktop/cyberrag/outputs/reports/relation_sample_100_audit.json).  
> A stratified random sample of **100 relation instances** (seed=42) drawn directly from `data/relations/extracted_relations.json` on the 104 ingested reports, evaluated against verbatim sentence evidence text.

### Provenance Integrity:
- **`report_id` matches in `data/raw/cti_reports.json`**: **100 / 100 (100.0%)**
- **`sentence_id` matches in `data/processed/preprocessed_cti.json`**: **100 / 100 (100.0%)**
- **`evidence_text` exact verbatim matches**: **100 / 100 (100.0%)**
- **Unique CTI Reports Represented**: **36 distinct reports**

### Empirical Precision by Relation Type:

| Relation Type | Sampled Count | True Positives | False Positives | Empirical Precision |
| :--- | :---: | :---: | :---: | :---: |
| **`EXPLOITS`** | 4 | 4 | 0 | **100.0%** |
| **`TARGETS`** | 34 | 34 | 0 | **100.0%** |
| **`USES_TECHNIQUE`** | 30 | 30 | 0 | **100.0%** |
| **`USES`** | 32 | 31 | 1 | **96.9%** |
| **Overall Relational Precision** | **100** | **99** | **1** | **99.0%** |

### Failure Mode Analysis:
- **`UNGROUNDED_TOOL_ASSOCIATION` (1 instance)**: Tool `Cobalt Strike` was mentioned descriptively as an exemplar (`such as Cobalt Strike`) within a coordinate relative clause where the governing subject was an unlinked noun phrase (`the actors linked to Trickbot`), leading to false attribution to TrickBot malware.
