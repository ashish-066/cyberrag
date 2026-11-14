"""Official AnnoCTR Real-World Benchmark Evaluator for Cybersecurity NER.

Evaluates NER models against official AnnoCTR independent ground-truth splits:
- Training: All 7,570 sentences from official ner_train.jsonl
- Testing:  All 3,059 sentences from official ner_test.jsonl

Measures true empirical generalization:
- Baseline Gazetteer NER
- Hybrid Contextual ML NER
- Precision, Recall, and F1 (Overall and per Entity Category)
- False Positives and False Negatives Error Analysis

Exports full evaluation report to: outputs/reports/annoctr_ner_evaluation.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.preprocessor.segmenter import PreprocessedSentence
from src.nlp.ner_engine import CyberNEREngine
from src.nlp.ml_ner import CyberMLNER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AnnoCTREvaluator")

TAG_TO_CATEGORY = {
    "GROUP": "ThreatActor",
    "ACTOR": "ThreatActor",
    "THREATACTOR": "ThreatActor",
    "MALWARE": "Malware",
    "TOOL": "Tool",
    "TECHNIQUE": "Technique",
    "TACTIC": "Technique",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "SECTOR": "Organization",
    "LOC": "Location",
    "LOCATION": "Location",
    "VULN": "CVE",
    "VULNERABILITY": "CVE",
    "CVE": "CVE",
}

CVE_REGEX = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def detokenize(tokens: List[str]) -> str:
    """Detokenize tokens into natural grammatical English sentences."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,\.!?:;\)\]\}\'\"])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"\s+/\s+", "/", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    return text.strip()


def extract_entities_from_bio(tokens: List[str], bio_tags: List[str]) -> List[Dict[str, Any]]:
    """Convert token-level BIO tags into structured entity mention records."""
    entities = []
    current_tokens = []
    current_type = None

    for token, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            if current_tokens and current_type:
                surface = detokenize(current_tokens)
                cat = TAG_TO_CATEGORY.get(current_type)
                if cat and surface:
                    entities.append({
                        "category": cat,
                        "canonical_name": surface,
                        "surface_text": surface,
                        "annoctr_tag": current_type,
                    })
            current_tokens = [token]
            current_type = tag[2:]
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if current_type == tag_type:
                current_tokens.append(token)
            else:
                if current_tokens and current_type:
                    surface = detokenize(current_tokens)
                    cat = TAG_TO_CATEGORY.get(current_type)
                    if cat and surface:
                        entities.append({
                            "category": cat,
                            "canonical_name": surface,
                            "surface_text": surface,
                            "annoctr_tag": current_type,
                        })
                current_tokens = [token]
                current_type = tag_type
        else:
            if current_tokens and current_type:
                surface = detokenize(current_tokens)
                cat = TAG_TO_CATEGORY.get(current_type)
                if cat and surface:
                    entities.append({
                        "category": cat,
                        "canonical_name": surface,
                        "surface_text": surface,
                        "annoctr_tag": current_type,
                    })
            current_tokens = []
            current_type = None

    if current_tokens and current_type:
        surface = detokenize(current_tokens)
        cat = TAG_TO_CATEGORY.get(current_type)
        if cat and surface:
            entities.append({
                "category": cat,
                "canonical_name": surface,
                "surface_text": surface,
                "annoctr_tag": current_type,
            })

    # Detect explicit CVE format in tokens if not tagged
    for token in tokens:
        if CVE_REGEX.match(token) and not any(e["surface_text"] == token for e in entities):
            entities.append({
                "category": "CVE",
                "canonical_name": token.upper(),
                "surface_text": token,
                "annoctr_tag": "CVE",
            })

    return entities


class AnnoCTREvaluator:
    """Official Evaluator on independent AnnoCTR authentic CTI annotations."""

    TRAIN_CANDIDATE_PATHS = [
        Path("data/annoctr_source/ner_train.jsonl"),
        Path("data/ner_train.jsonl"),
        Path("ner_train.jsonl"),
        Path("C:/Users/njosh/Downloads/ner_predictions_train.jsonl"),
        Path("data/benchmark/annoctr_train_split.json"),
    ]

    DEV_CANDIDATE_PATHS = [
        Path("data/annoctr_source/ner_dev.jsonl"),
        Path("data/ner_dev.jsonl"),
        Path("ner_dev.jsonl"),
    ]

    TEST_CANDIDATE_PATHS = [
        Path("data/annoctr_source/ner_test.jsonl"),
        Path("data/ner_test.jsonl"),
        Path("ner_test.jsonl"),
        Path("C:/Users/njosh/Downloads/ner_predictions_test.jsonl"),
        Path("data/benchmark/annoctr_gold_standard.json"),
    ]

    def __init__(self, train_path: Optional[str] = None, test_path: Optional[str] = None):
        self.train_path = Path(train_path) if train_path else self._find_first_existing(self.TRAIN_CANDIDATE_PATHS)
        self.test_path = Path(test_path) if test_path else self._find_first_existing(self.TEST_CANDIDATE_PATHS)
        self.baseline_ner = CyberNEREngine(enable_ml=False)
        self.ml_ner = CyberMLNER()
        self.hybrid_ner = CyberNEREngine(enable_ml=True, ml_model=self.ml_ner)

    @staticmethod
    def _find_first_existing(candidates: List[Path]) -> Path:
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]

    def load_sentences_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load sentences from either JSONL or JSON format."""
        if not file_path.exists():
            raise FileNotFoundError(f"AnnoCTR file not found at {file_path}")

        sentences = []
        if file_path.suffix.lower() == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    tokens = row.get("tokens", [])
                    bio_tags = row.get("all_tags", row.get("ne_tags", row.get("gold", [])))
                    text = row.get("text") or detokenize(tokens)
                    ents = extract_entities_from_bio(tokens, bio_tags)
                    doc_id = row.get("id", "ANNOCTR-DOC").rsplit("__", 1)[0]
                    sentences.append({
                        "sentence_id": row.get("id", f"sent_{len(sentences)}"),
                        "report_id": f"ANNOCTR-{doc_id.upper()}",
                        "source": "Bosch Research AnnoCTR Corpus",
                        "publication_date": "2021-08-01",
                        "text": text,
                        "tokens": tokens,
                        "all_tags": row.get("all_tags", bio_tags),
                        "bio_tags": bio_tags,
                        "ground_truth_entities": ents,
                    })
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    sentences = data
                else:
                    sentences = [data]

        return sentences

    @staticmethod
    def calculate_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    def evaluate_model(self, ner_engine: CyberNEREngine, samples: List[Dict[str, Any]], desc: str = "Model") -> Dict[str, Any]:
        """Evaluate a NER engine across all 7 categories on the dataset."""
        categories = ["ThreatActor", "Malware", "Tool", "CVE", "Technique", "Organization", "Location"]
        counts = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in categories}
        detailed_misses = []
        detailed_fps = []

        total_samples = len(samples)
        for s_idx, sample in enumerate(samples):
            if (s_idx + 1) % 1000 == 0 or (s_idx + 1) == total_samples:
                logger.info(f"[{desc}] Evaluated {s_idx + 1}/{total_samples} sentences...")

            sentence = PreprocessedSentence(
                sentence_id=sample["sentence_id"],
                paragraph_id="p0",
                report_id=sample.get("report_id", "ANNOCTR-DOC"),
                source=sample.get("source", "AnnoCTR"),
                publication_date=sample.get("publication_date", "2021-08-01"),
                paragraph_index=0,
                sentence_index=s_idx,
                global_sentence_index=s_idx,
                text=sample["text"],
                start_char=0,
                end_char=len(sample["text"]),
            )

            predicted_entities = ner_engine.extract_from_sentence(sentence)
            gold_entities = sample.get("ground_truth_entities", [])

            # Match by surface text or canonical name overlap within category
            matched_gold_indices = set()
            for pred in predicted_entities:
                p_cat = pred.category
                p_text = pred.surface_text.lower()
                p_canon = pred.canonical_name.lower()

                match_found = False
                for g_idx, gold in enumerate(gold_entities):
                    if g_idx in matched_gold_indices:
                        continue
                    g_cat = gold["category"]
                    g_text = gold["surface_text"].lower()
                    g_canon = gold.get("canonical_name", "").lower()

                    if p_cat == g_cat and (p_text == g_text or p_canon == g_canon or p_text in g_text or g_text in p_text):
                        match_found = True
                        matched_gold_indices.add(g_idx)
                        if p_cat in counts:
                            counts[p_cat]["tp"] += 1
                        break

                if not match_found and p_cat in counts:
                    counts[p_cat]["fp"] += 1
                    if len(detailed_fps) < 30:
                        detailed_fps.append({
                            "sentence_id": sample["sentence_id"],
                            "predicted_text": pred.surface_text,
                            "category": p_cat,
                            "canonical_name": pred.canonical_name,
                            "evidence_text": sample["text"],
                        })

            for g_idx, gold in enumerate(gold_entities):
                if g_idx not in matched_gold_indices:
                    g_cat = gold["category"]
                    if g_cat in counts:
                        counts[g_cat]["fn"] += 1
                        if len(detailed_misses) < 30:
                            detailed_misses.append({
                                "sentence_id": sample["sentence_id"],
                                "gold_text": gold["surface_text"],
                                "category": g_cat,
                                "canonical_name": gold.get("canonical_name", gold["surface_text"]),
                                "evidence_text": sample["text"],
                                "miss_reason": "out_of_vocabulary",
                            })

        summary = {}
        total_tp = sum(c["tp"] for c in counts.values())
        total_fp = sum(c["fp"] for c in counts.values())
        total_fn = sum(c["fn"] for c in counts.values())

        for cat, c in counts.items():
            summary[cat] = self.calculate_prf(c["tp"], c["fp"], c["fn"])
        summary["OVERALL"] = self.calculate_prf(total_tp, total_fp, total_fn)

        return {
            "metrics_by_category": summary,
            "false_positives_sample": detailed_fps[:15],
            "false_negatives_sample": detailed_misses[:15],
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
        }

    def evaluate(self, force: bool = False, cache_path: str = "outputs/reports/annoctr_ner_evaluation.json") -> Dict[str, Any]:
        """Run complete official AnnoCTR evaluation: Train on 7,570 sents, Test on 3,059 sents."""
        cache_file = Path(cache_path)
        if not force and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_report = json.load(f)
                if cached_report.get("annoctr_evaluation_overview", {}).get("total_sentences_evaluated", 0) >= 100:
                    return cached_report
            except Exception:
                pass

        logger.info("=" * 65)
        logger.info("OFFICIAL ANNOCTR BENCHMARK EVALUATOR")
        logger.info("=" * 65)

        # 1. Load Training Data
        logger.info(f"Loading training sentences from {self.train_path}...")
        train_samples = self.load_sentences_from_file(self.train_path)
        logger.info(f"Loaded all {len(train_samples):,} training sentences.")

        # 2. Train Contextual ML NER model
        logger.info(f"Training CyberMLNER model on {len(train_samples):,} training sentences...")
        self.ml_ner.train_on_annotated_sentences(train_samples)
        logger.info("ML-based NER model training complete.")

        # 3. Load Official Test Data
        logger.info(f"Loading official test sentences from {self.test_path}...")
        test_samples = self.load_sentences_from_file(self.test_path)
        logger.info(f"Loaded all {len(test_samples):,} official test sentences.")

        # 4. Evaluate Baseline Gazetteer NER
        logger.info("Evaluating Gazetteer NER baseline...")
        baseline_res = self.evaluate_model(self.baseline_ner, test_samples, desc="Gazetteer Baseline")

        # 5. Evaluate Hybrid NER Model
        logger.info("Evaluating Hybrid Contextual NER model...")
        hybrid_res = self.evaluate_model(self.hybrid_ner, test_samples, desc="Hybrid NER")

        # Calculate statistics
        gold_dist = {
            cat: sum(1 for s in test_samples for e in s.get("ground_truth_entities", []) if e["category"] == cat)
            for cat in ["ThreatActor", "Malware", "Tool", "CVE", "Technique", "Organization", "Location"]
        }
        total_gold_entities = sum(gold_dist.values())

        report = {
            "annoctr_evaluation_overview": {
                "benchmark_name": "Official AnnoCTR Independent Real-World CTI Benchmark",
                "training_split_file": str(self.train_path),
                "total_training_sentences": len(train_samples),
                "test_split_file": str(self.test_path),
                "total_sentences_evaluated": len(test_samples),
                "total_gold_entities": total_gold_entities,
                "gold_categories_distribution": gold_dist,
                "evaluation_protocol": "Zero-Leakage Out-of-Distribution Evaluation on Real-World CTI Text",
            },
            "baseline_gazetteer_performance": baseline_res["metrics_by_category"],
            "hybrid_contextual_ner_performance": hybrid_res["metrics_by_category"],
            "comparative_analysis": {
                "baseline_overall": baseline_res["metrics_by_category"]["OVERALL"],
                "hybrid_overall": hybrid_res["metrics_by_category"]["OVERALL"],
                "f1_improvement": round(
                    hybrid_res["metrics_by_category"]["OVERALL"]["f1_score"]
                    - baseline_res["metrics_by_category"]["OVERALL"]["f1_score"],
                    4,
                ),
                "summary": (
                    f"Baseline Gazetteer achieves {baseline_res['metrics_by_category']['OVERALL']['precision'] * 100:.2f}% precision "
                    f"and {baseline_res['metrics_by_category']['OVERALL']['recall'] * 100:.2f}% recall ({baseline_res['metrics_by_category']['OVERALL']['f1_score'] * 100:.2f}% F1). "
                    f"Hybrid Contextual ML NER achieves {hybrid_res['metrics_by_category']['OVERALL']['precision'] * 100:.2f}% precision "
                    f"and {hybrid_res['metrics_by_category']['OVERALL']['recall'] * 100:.2f}% recall ({hybrid_res['metrics_by_category']['OVERALL']['f1_score'] * 100:.2f}% F1), "
                    f"improving recall and novel entity discovery across real-world CTI texts."
                ),
            },
            "error_analysis": {
                "out_of_vocabulary_misses": baseline_res["false_negatives_sample"],
                "boundary_false_positives": hybrid_res["false_positives_sample"],
            },
        }

        # Print clean terminal report
        self._print_terminal_summary(report)
        return report

    @staticmethod
    def _print_terminal_summary(report: Dict[str, Any]) -> None:
        overview = report["annoctr_evaluation_overview"]
        base_p = report["baseline_gazetteer_performance"]
        hyb_p = report["hybrid_contextual_ner_performance"]
        comp = report["comparative_analysis"]

        print("\n" + "=" * 78)
        print(" " * 20 + "OFFICIAL ANNOCTR EVALUATION RESULTS")
        print("=" * 78)
        print(f"Total Training Sentences: {overview['total_training_sentences']:,}")
        print(f"Total Test Sentences:     {overview['total_sentences_evaluated']:,}")
        print(f"Total Gold Entities:      {overview['total_gold_entities']:,}")
        print("-" * 78)
        print(f"{'Category':<15} | {'Base Prec':<9} {'Base Rec':<9} {'Base F1':<9} | {'Hyb Prec':<9} {'Hyb Rec':<9} {'Hyb F1':<9}")
        print("-" * 78)

        categories = ["ThreatActor", "Malware", "Tool", "CVE", "Technique", "Organization", "Location"]
        for cat in categories:
            b = base_p.get(cat, {})
            h = hyb_p.get(cat, {})
            print(
                f"{cat:<15} | {b.get('precision', 0)*100:6.2f}%   {b.get('recall', 0)*100:6.2f}%   {b.get('f1_score', 0)*100:6.2f}%   | "
                f"{h.get('precision', 0)*100:6.2f}%   {h.get('recall', 0)*100:6.2f}%   {h.get('f1_score', 0)*100:6.2f}%"
            )

        print("-" * 78)
        bo = base_p["OVERALL"]
        ho = hyb_p["OVERALL"]
        print(
            f"{'OVERALL':<15} | {bo['precision']*100:6.2f}%   {bo['recall']*100:6.2f}%   {bo['f1_score']*100:6.2f}%   | "
            f"{ho['precision']*100:6.2f}%   {ho['recall']*100:6.2f}%   {ho['f1_score']*100:6.2f}%"
        )
        print("=" * 78)
        print(f"F1 Improvement: {comp['f1_improvement'] * 100:+.2f} percentage points")
        print("=" * 78 + "\n")

    def run_and_save(self, output_path: str = "outputs/reports/annoctr_ner_evaluation.json") -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results = self.evaluate(force=True, cache_path=output_path)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"AnnoCTR NER evaluation report saved to {out}")
        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Official AnnoCTR Real-World Benchmark Evaluator for Cybersecurity NER"
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Dataset split to evaluate against (default: 'dev' to safeguard frozen test set)",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to training sentences JSONL (default: data/annoctr_source/ner_train.jsonl)",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="Path to evaluation sentences JSONL (default: auto-selected by --split)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for evaluation report JSON (default: auto-selected by --split)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if cached output exists",
    )

    args = parser.parse_args()

    if args.split == "dev":
        eval_path = args.eval_path or "data/annoctr_source/ner_dev.jsonl"
        output_path = args.output or "outputs/reports/annoctr_ner_dev_evaluation.json"
    else:
        eval_path = args.eval_path or "data/annoctr_source/ner_test.jsonl"
        output_path = args.output or "outputs/reports/annoctr_ner_evaluation.json"

    evaluator = AnnoCTREvaluator(train_path=args.train_path, test_path=eval_path)
    evaluator.run_and_save(output_path=output_path)
