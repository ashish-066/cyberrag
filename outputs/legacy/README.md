# Legacy Artefacts (Archived Prototype Benchmarks)

> [!WARNING]
> **DO NOT CITE THESE METRICS AS EMPIRICAL EVIDENCE.**
>
> The files in this directory (`evaluate_llm_rag_ablation.py` and `llm_rag_retrieval_ablation.json`) represent exploratory ablation experiments from an earlier prototype (v5) and are preserved solely for architectural transparency and archival record.
>
> **Methodological Circularity**:
> Rigorous internal methodological review determined that this exploratory ablation contained an architectural circularity: the hand-curated alias dictionary (`KNOWN_ALIASES`) used to generate the ground-truth retrieval labels was simultaneously supplied to the expanded retrieval loop. Consequently, the observed retrieval gains do **not** constitute an unbiased, generalizable measurement of retrieval performance.
>
> In addition, `evaluate_llm_rag_ablation.py` imports deprecated prototype components (`src.llm.query_planner`, `src.llm.query_expander`) that are not part of the production pipeline, and the script is intentionally not executable against the current codebase.
>
> Sound, non-circular relation evaluation against independent ground truth (such as AZERG) is documented in the main README as planned future work.
