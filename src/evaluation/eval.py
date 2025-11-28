from evaluate import load
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.file_utils import save_to_json

class EvaluationMetrics:
    def __init__(self):
        """
        Initialize the metrics loaders.
        """
        self.bleu = load("bleu")
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
        self.f1_metric = load("f1")
        self.squad_metric = load("squad")
    
    @staticmethod
    def exact_match(pred, ref):
        """
        Compute exact match score between prediction and reference.
        
        Args:
            pred (str): Predicted text.
            ref (str): Reference text.
        
        Returns:
            int: 1 if exact match, 0 otherwise.
        """
        return int(pred.strip() == ref.strip())
    

    def f1_score(self, pred, ref):
        """
        Compute F1 score between prediction and reference.
        Args:
            pred (str): Predicted text.
            ref (str): Reference text.
        
        Returns:
            float: F1 score.
        """
        return self.f1_metric.compute(predictions=[pred], references=[ref])["f1"]


    def compute_metrics(self, predictions):
        """
        Compute evaluation metrics for a list of predictions and references.

        Args:
            predictions (list of tuple): List of (prediction, reference) pairs.

        Returns:
            dict: Dictionary containing BLEU, ROUGE-L, Exact Match, BERTScore, and F1+EM metrics.
        """
        preds = [p[0] for p in predictions]
        golds = [p[1] for p in predictions]

        # Format for squad_metric
        squad_predictions = [
            {"id": str(i), "prediction_text": preds[i]}
            for i in range(len(preds))
        ]

        squad_references = [
            {"id": str(i), "answers": {"text": [golds[i]], "answer_start": [0]}}
            for i in range(len(golds))
        ]

        return {
            "BLEU": self.bleu.compute(predictions=preds, references=[[g] for g in golds])["bleu"],
            "ROUGE-L": self.rouge.compute(predictions=preds, references=golds)["rougeL"],
            "Exact_Match": sum(self.exact_match(p, g) for p, g in predictions) / len(predictions),
            "BERTScore": np.mean(self.bertscore.compute(predictions=preds, references=golds, lang="en")["f1"]),
            "F1+EM": self.squad_metric.compute(
                predictions=squad_predictions,
                references=squad_references
            )
        }



    def save_metrics_to_json(self, metrics, file_path):
        """
        Save metrics to a JSON file.

        Args:
            metrics (dict): Dictionary containing evaluation metrics.
            file_path (str): Path to save the JSON file.
        """
        try:
            save_to_json(metrics, file_path)
            print(f"Metrics saved to {file_path}")
        except Exception as e:
            print(f"Error saving metrics to {file_path}: {e}")

if __name__ == "__main__":
    predictions_base = [("prediction1_base", "reference1"), ("prediction2_base", "reference2")]
    predictions_ft = [("prediction1_ft", "reference1"), ("prediction2_ft", "reference2")]

    evaluator = EvaluationMetrics()

    print("=== Evaluating Base Model ===")
    metrics_base = evaluator.compute_metrics(predictions_base)
    print(metrics_base)

    print("=== Evaluating Fine-tuned Model ===")
    metrics_ft = evaluator.compute_metrics(predictions_ft)
    print(metrics_ft)

    # Save metrics to JSON
    evaluator.save_metrics_to_json({"base": metrics_base, "fine_tuned": metrics_ft}, "eval_comparison.json")