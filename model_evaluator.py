#model_evaluator.py


import os
import json
import time
import numpy as np
import sacrebleu
from datetime import datetime


class ModelEvaluator:

    def __init__(self, strict_mode=False):
        
        self.results_dir = "results/evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.strict_mode = strict_mode

    #  BLEU calculation
    def calculate_bleu(self, candidate, reference):
        """Calculate BLEU score between candidate and reference (0–1 scale)."""
        if not candidate or not reference:
            return 0.0
        try:
            bleu = sacrebleu.corpus_bleu([candidate], [[reference]])
            return float(bleu.score / 100.0)
        except Exception as e:
            print(f" BLEU calculation error: {e}")
            return 0.0

    #  Precision, Recall, and F1 computation
    def calculate_accuracy(self, translated_text, reference_text):
        """
        Calculate precision, recall, and F1 using word overlap.
        Returns dict with metrics and threshold flag (>=95%).
        """
        if not translated_text or not reference_text:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "meets_threshold": False}

        try:
            translated_words = set(translated_text.lower().split())
            reference_words = set(reference_text.lower().split())

            if not translated_words or not reference_words:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "meets_threshold": False}

            common_words = translated_words.intersection(reference_words)
            precision = len(common_words) / len(translated_words)
            recall = len(common_words) / len(reference_words)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            meets_threshold = all(m >= 0.95 for m in [precision, recall, f1])

            return {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "meets_threshold": meets_threshold
            }

        except Exception as e:
            print(f" Accuracy calculation error: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "meets_threshold": False}

    def measure_latency(self, fn, *args, **kwargs):
        """Measure average latency (in seconds) over multiple runs."""
        n = kwargs.pop("runs", 3)
        times = []
        res = None
        for _ in range(n):
            t0 = time.perf_counter()
            res = fn(*args, **kwargs)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg_time = float(np.mean(times)) if times else 0.0
        return avg_time, res

    def evaluate_translation_quality(self, translator):
        """
        Evaluate translation model using BLEU, precision, recall, F1, and latency.
        translator must implement: translate_text(text, source_lang, target_lang)
        """
        print("\n COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)

        # Test corpus
        test_samples = [
            {
                "en": "Welcome to the live sports commentary",
                "hi": "लाइव स्पोर्ट्स कमेंट्री में आपका स्वागत है",
                "es": "Bienvenido al comentario deportivo en vivo",
                "fr": "Bienvenue au commentaire sportif en direct"
            },
            {
                "en": "What an amazing goal by the team",
                "hi": "टीम द्वारा क्या अद्भुत गोल है",
                "es": "Qué gol tan increíble del equipo",
                "fr": "Quel but incroyable de l'équipe"
            }
        ]

        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "test_cases": len(test_samples),
            "language_pairs": {},
            "overall_metrics": {}
        }

        all_bleu_scores, all_f1_scores, all_latencies = [], [], []
        successful_translations = 0
        total_translations = 0

        print("\n EVALUATING TRANSLATIONS:\n")

        for source_lang in ["en"]:
            for target_lang in ["hi", "es", "fr"]:
                pair_key = f"{source_lang}-{target_lang}"
                evaluation_results["language_pairs"][pair_key] = {
                    "bleu_scores": [],
                    "precision_scores": [],
                    "recall_scores": [],
                    "f1_scores": [],
                    "latency_scores": [],
                    "successful_translations": 0
                }

                for i, sample in enumerate(test_samples):
                    source_text = sample.get(source_lang)
                    reference_text = sample.get(target_lang)
                    if not source_text or not reference_text:
                        continue

                    total_translations += 1
                    latency, translated = self.measure_latency(
                        translator.translate_text, source_text, source_lang, target_lang
                    )

                    if translated:
                        successful_translations += 1

                        bleu_score = self.calculate_bleu(translated, reference_text)
                        acc = self.calculate_accuracy(translated, reference_text)

                        pair_metrics = evaluation_results["language_pairs"][pair_key]
                        pair_metrics["bleu_scores"].append(bleu_score)
                        pair_metrics["precision_scores"].append(acc["precision"])
                        pair_metrics["recall_scores"].append(acc["recall"])
                        pair_metrics["f1_scores"].append(acc["f1"])
                        pair_metrics["latency_scores"].append(latency)
                        pair_metrics["successful_translations"] += 1

                        all_bleu_scores.append(bleu_score)
                        all_f1_scores.append(acc["f1"])
                        all_latencies.append(latency)

                        print(
                            f" {pair_key.upper()} | Sample {i+1} | "
                            f"BLEU={bleu_score:.3f} | "
                            f"P={acc['precision']:.2f} R={acc['recall']:.2f} F1={acc['f1']:.2f} | "
                            f"Latency={latency:.3f}s"
                        )

                        # Warn or fail if below threshold
                        if not acc["meets_threshold"]:
                            print(f"  {pair_key.upper()} | Below 95% threshold "
                                  f"(P={acc['precision']:.2f}, R={acc['recall']:.2f}, F1={acc['f1']:.2f})")
                            if self.strict_mode:
                                raise AssertionError(
                                    f" Translation quality below 95% for {pair_key.upper()} Sample {i+1}"
                                )
                    else:
                        print(f" {pair_key.upper()} | Sample {i+1}: No translation returned")

        # Compute averages
        if total_translations > 0:
            evaluation_results["overall_metrics"] = {
                "average_bleu_score": round(float(np.mean(all_bleu_scores)), 3) if all_bleu_scores else 0.0,
                "average_f1_score": round(float(np.mean(all_f1_scores)), 3) if all_f1_scores else 0.0,
                "average_latency": round(float(np.mean(all_latencies)), 3) if all_latencies else 0.0,
                "total_translations": total_translations,
                "successful_translations": successful_translations,
                "success_rate": round(successful_translations / total_translations, 3)
                if total_translations > 0 else 0.0
            }

        self._display_evaluation_results(evaluation_results)
        self._save_evaluation_results(evaluation_results)
        return evaluation_results

    #  Backward compatibility wrapper for Milestone-3
    def run_comprehensive_evaluation(self, translator):
        """Compatibility alias for translation_pipeline.py"""
        return self.evaluate_translation_quality(translator)

    def _display_evaluation_results(self, results):
        """Pretty-print evaluation results"""
        print("\n" + "=" * 70)
        print(" FINAL EVALUATION RESULTS")
        print("=" * 70)

        overall = results.get("overall_metrics", {})

        print(f"\n OVERALL PERFORMANCE:")
        print(f"    Average BLEU Score: {overall.get('average_bleu_score', 0):.3f}")
        print(f"    Average F1 Score: {overall.get('average_f1_score', 0):.3f}")
        print(f"    Average Latency: {overall.get('average_latency', 0):.3f}s")
        print(f"    Success Rate: {overall.get('success_rate', 0):.1%}")
        print(f"    Total Tests: {overall.get('total_translations', 0)}")

        print(f"\n DETAILED PERFORMANCE BY LANGUAGE PAIR:")
        for pair, metrics in results.get("language_pairs", {}).items():
            if metrics.get("successful_translations", 0) > 0:
                avg_bleu = np.mean(metrics["bleu_scores"]) if metrics["bleu_scores"] else 0
                avg_f1 = np.mean(metrics["f1_scores"]) if metrics["f1_scores"] else 0
                avg_latency = np.mean(metrics["latency_scores"]) if metrics["latency_scores"] else 0
                print(f"   {pair.upper():8} | BLEU: {avg_bleu:.3f} | F1: {avg_f1:.3f} | Latency: {avg_latency:.3f}s")

    def _save_evaluation_results(self, results):
        """Save evaluation results as JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_file = os.path.join(self.results_dir, f"evaluation_{timestamp}.json")
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n Evaluation results saved to: {eval_file}")
        except Exception as e:
            print(f" Error saving evaluation results: {e}")


# Backward compatibility alias
EnhancedModelEvaluator = ModelEvaluator
