# CyberRAG Benchmark Specification: Synthetic Controlled Challenge Set

> **Benchmark Name**: CyberRAG-50 Synthetic Controlled Challenge Benchmark  
> **Benchmark Type**: Synthetic Rule-Stress & Adversarial Challenge Set  
> **Total Sentences**: 50  
> **Ground-Truth Entities**: 155 mentions (7 categories)  
> **Ground-Truth Relations**: 93 triples (4 types)  

---

## 1. Benchmark Origin & Taxonomy

The CyberRAG-50 benchmark is an **author-constructed synthetic challenge set** designed specifically to stress-test extraction rules and machine-learning boundaries under controlled syntactic conditions. Rather than containing unfiltered raw CTI paragraphs (which often exhibit severe narrative dilution), the benchmark evaluates whether the NLP pipeline can handle syntactically complex and adversarial constructions:

1. **Paraphrased CTI Propositions (`paraphrased_cti`) — 31 Sentences (62%)**:
   - Synthesized from authentic CTI advisories published by CISA, Mandiant, CrowdStrike, SentinelOne, and Unit 42.
   - Condenses multi-clause findings into precise cybersecurity assertions while preserving canonical threat actor names, CVEs, techniques, and victim entities.
2. **Constructed Challenge Sentences (`constructed_challenge`) — 19 Sentences (38%)**:
   - Artificially engineered stress-test examples targeting common failure modes in rule-based and machine-learning extractors:
     - **Passive Voice Inversion (5 sentences)**: Evaluates whether the extractor correctly inverts subject-object roles when the patient precedes the verb (e.g. `[CVE] was actively exploited by [Actor]`).
     - **Localized Negation (4 sentences)**: Tests affirmative relation suppression in negative contexts (e.g. `Actor did not deploy Malware during intrusions against Target`).
     - **Vendor Attribution / Reporting Distractors (4 sentences)**: Tests suppression of false attacker roles on intelligence publishers (e.g. `CrowdStrike analysts published an overview of Russian campaigns...`).
     - **Multi-Target Infrastructure (2 sentences)**: Evaluates extraction of multiple parallel victim organizations/locations in a single clause.
     - **Multi-Actor Clause Boundaries (1 sentence)**: Tests intervening actor blocking across coordinate and subordinate clauses.
     - **Negative Control / Distractor (1 sentence)**: Verifies that sentences mentioning multiple threat actors without attack actions yield zero false relations.
     - **Multi-Tool & Technique Delivery (2 sentences)**: Evaluates disambiguation between tool execution and technique delivery.

---

## 2. Annotation Methodology & Integrity

- **Benchmark Construction**: The 50 benchmark sentences were curated and labeled by the pipeline developer to establish an automated unit and integration regression test suite.
- **Scope & Limitations**:
  - This benchmark is a **synthetic unit/regression suite**, not a multi-annotator sociological ground truth. Previous claims of dual-annotator adjudication and Cohen's $\kappa$ have been removed to ensure complete academic and factual integrity.
  - Fair evaluation against real-world, out-of-distribution text is evaluated separately on authentic, multi-paragraph CTI reports.
- **Quality Adjudication Fixes**:
  - In `GOLD::050` (*"APT29 targeted Ukrainian government organizations to collect intelligence on geopolitical strategies."*), the target is correctly labeled as `Ukrainian Government Organizations` (`Organization`), and the unsupported spurious relation `APT29 TARGETS Ukraine` has been removed.

---

## 3. Schema Coverage

### Entity Categories (7 Categories, 155 Mentions)
- `ThreatActor`: 48
- `Organization`: 25
- `Location`: 27
- `Tool`: 22
- `Malware`: 11
- `CVE`: 11
- `Technique`: 11

### Relationship Types (4 Relations, 93 Triples)
- `TARGETS`: 42
- `USES`: 31
- `EXPLOITS`: 10
- `USES_TECHNIQUE`: 10
