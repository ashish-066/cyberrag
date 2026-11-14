"""Master Pipeline Runner for Person 1: Data + Cyber NLP + Knowledge Graph.

Executes end-to-end:
1. Dataset loading & validation
2. Text cleaning & paragraph/sentence segmentation preserving provenance
3. Cybersecurity NER & entity normalization/linking
4. Typed relationship extraction (USES, TARGETS, EXPLOITS, USES_TECHNIQUE)
5. Cyber Knowledge Graph construction & exports (JSON, GraphML, Cypher)
6. Interactive HTML explorer and high-res static visual figures
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset.loader import CTIDatasetLoader
from src.dataset.downloader import CTIDatasetDownloader
from src.preprocessor.segmenter import CTISegmenter
from src.nlp.ner_engine import CyberNEREngine
from src.nlp.relation_extractor import CyberRelationExtractor
from src.nlp.evaluator import BenchmarkEvaluator
from src.graph.builder import CyberKnowledgeGraphBuilder
from src.graph.exporter import CyberKnowledgeGraphExporter
from src.graph.analytics import CyberKnowledgeGraphAnalytics
from src.visualization.interactive_html import InteractiveGraphVisualizer
from src.visualization.static_plotter import StaticGraphPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CyberRAG.Pipeline")


def run_pipeline(
    raw_data_dir: str = "data/raw",
    processed_output: str = "data/processed/preprocessed_cti.json",
    entities_output: str = "data/entities/extracted_entities.json",
    relations_output: str = "data/relations/extracted_relations.json",
    kg_json_output: str = "data/knowledge_graph/cyber_kg.json",
    kg_graphml_output: str = "data/knowledge_graph/cyber_kg.graphml",
    kg_cypher_output: str = "data/knowledge_graph/import_kg.cypher",
    analytics_output: str = "outputs/reports/cti_kg_summary.json",
    html_output: str = "outputs/visualizations/cyber_knowledge_graph.html",
    skip_static_plots: bool = False,
):
    """Run full CTI to Cyber Knowledge Graph pipeline."""
    logger.info("=" * 65)
    logger.info("Starting CyberRAG Knowledge Graph Pipeline")
    logger.info("=" * 65)

    # -------------------------------------------------------------
    # Step 1: Ensure and Load Dataset
    # -------------------------------------------------------------
    logger.info("[Step 1/6] Ingesting and Loading Raw CTI Reports...")
    downloader = CTIDatasetDownloader(output_dir=raw_data_dir)
    downloader.ensure_dataset()

    loader = CTIDatasetLoader(raw_data_dir=raw_data_dir)
    reports = loader.load_all_raw()
    if not reports:
        logger.error(f"No CTI reports found in {raw_data_dir}. Exiting.")
        return 1

    logger.info(f"Loaded {len(reports)} raw CTI reports successfully.")

    # -------------------------------------------------------------
    # Step 2: Clean and Preprocess CTI Reports
    # Preserve: report ID, source, paragraph/sentence, publication date
    # -------------------------------------------------------------
    logger.info("[Step 2/6] Cleaning and Preprocessing CTI Reports...")
    segmenter = CTISegmenter()
    preprocessed_reports = segmenter.process_all(reports)

    total_paras = sum(r.total_paragraphs for r in preprocessed_reports)
    total_sents = sum(r.total_sentences for r in preprocessed_reports)
    logger.info(f"Preprocessed {len(preprocessed_reports)} reports -> {total_paras} paragraphs, {total_sents} sentences.")

    # Save preprocessed reports
    Path(processed_output).parent.mkdir(parents=True, exist_ok=True)
    with open(processed_output, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in preprocessed_reports], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved preprocessed reports to {processed_output}")

    # -------------------------------------------------------------
    # Step 3: Cybersecurity Named Entity Recognition & Linking
    # -------------------------------------------------------------
    logger.info("[Step 3/6] Running Cybersecurity NER & Entity Linking...")
    ner_engine = CyberNEREngine()
    extracted_entities = ner_engine.extract_all(preprocessed_reports)

    # Entity category stats
    category_counts = {}
    for e in extracted_entities:
        category_counts[e.category] = category_counts.get(e.category, 0) + 1

    logger.info(f"Extracted {len(extracted_entities)} entity mentions:")
    for cat, cnt in sorted(category_counts.items()):
        logger.info(f"  - {cat}: {cnt}")

    # Save extracted entities
    Path(entities_output).parent.mkdir(parents=True, exist_ok=True)
    with open(entities_output, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in extracted_entities], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved extracted entities to {entities_output}")

    # -------------------------------------------------------------
    # Step 4: Extract Typed Relationships
    # (USES, TARGETS, EXPLOITS, USES_TECHNIQUE)
    # -------------------------------------------------------------
    logger.info("[Step 4/6] Extracting Typed Relationships with Provenance...")
    relation_extractor = CyberRelationExtractor(ner_engine=ner_engine)
    _, extracted_relations = relation_extractor.extract_all(preprocessed_reports)

    # Relation type stats
    relation_counts = {}
    for r in extracted_relations:
        relation_counts[r.relation_type] = relation_counts.get(r.relation_type, 0) + 1

    logger.info(f"Extracted {len(extracted_relations)} relationship instances:")
    for r_type, cnt in sorted(relation_counts.items()):
        logger.info(f"  - {r_type}: {cnt}")

    # Save extracted relations
    Path(relations_output).parent.mkdir(parents=True, exist_ok=True)
    with open(relations_output, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in extracted_relations], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved extracted relations to {relations_output}")

    # -------------------------------------------------------------
    # Step 5: Construct Cyber Knowledge Graph
    # -------------------------------------------------------------
    logger.info("[Step 5/6] Constructing Initial Cyber Knowledge Graph...")
    builder = CyberKnowledgeGraphBuilder()
    kg = builder.build(extracted_entities, extracted_relations)

    logger.info(f"Knowledge Graph constructed: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")

    # Export graph
    exporter = CyberKnowledgeGraphExporter(kg)
    exporter.export_json(kg_json_output)
    exporter.export_graphml(kg_graphml_output)
    exporter.export_cypher(kg_cypher_output)

    # Analytics & Summary
    analytics = CyberKnowledgeGraphAnalytics(kg)
    analytics.export_summary(analytics_output)

    # Human-annotated Gold Standard Evaluation Benchmark
    logger.info("Evaluating Pipeline on Gold Standard Benchmark...")
    evaluator = BenchmarkEvaluator()
    eval_report_path = "outputs/reports/evaluation_benchmark.json"
    evaluator.run_and_save(eval_report_path)
    logger.info(f"Saved evaluation benchmark report to {eval_report_path}")

    # -------------------------------------------------------------
    # Step 6: Produce Graph Visualizations
    # -------------------------------------------------------------
    logger.info("[Step 6/6] Generating Interactive & Static Graph Visualizations...")
    visualizer = InteractiveGraphVisualizer(kg)
    visualizer.generate_html(html_output)
    logger.info(f"Interactive HTML Visualization available at {html_output}")

    if not skip_static_plots:
        try:
            plotter = StaticGraphPlotter(kg)
            plots = plotter.generate_all_plots()
            logger.info(f"Static plot figures created: {[p.name for p in plots]}")
        except Exception as e:
            logger.warning(f"Could not render static matplotlib plots: {e}")

    logger.info("=" * 65)
    logger.info("CyberRAG Pipeline Completed Successfully!")
    logger.info("=" * 65)
    return 0


def main():
    parser = argparse.ArgumentParser(description="CyberRAG CTI Knowledge Graph Pipeline Runner")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw CTI reports directory")
    parser.add_argument("--skip-static", action="store_true", help="Skip static matplotlib plot generation")
    args = parser.parse_args()

    exit_code = run_pipeline(
        raw_data_dir=args.raw_dir,
        skip_static_plots=args.skip_static,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
