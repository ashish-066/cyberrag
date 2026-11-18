# Legacy artefacts (not part of the runtime)

`evaluate_llm_rag_ablation.py` and `llm_rag_retrieval_ablation.json` come from v5.
The ablation was **circular**: the same hand-written alias table (`KNOWN_ALIASES`)
that produced the retrieval gold labels was also given to the "expanded" system,
so the reported gains do not measure retrieval quality. The script imports modules
that no longer exist (`src.llm.query_planner`, `src.llm.query_expander`) and will not run.
Kept only for reference; do not cite the numbers. A sound ablation belongs to Stage 2.
