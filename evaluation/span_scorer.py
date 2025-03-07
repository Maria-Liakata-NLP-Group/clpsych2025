import torch
from transformers import AutoTokenizer
from bert_score import BERTScorer
import numpy as np
from typing import List


class SpanScorer:
    def __init__(
        self,
        model_name: str = "microsoft/deberta-xlarge-mnli",
        rescale_with_baseline: bool = True,
    ):
        self.task = "A.1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.rescale_with_baseline = rescale_with_baseline
        self.scorer = BERTScorer(
            model_type=self.model_name,
            lang="en",
            device=self.device,
            rescale_with_baseline=self.rescale_with_baseline,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=False
        )

    def score_empty_predictions(self):
        """Return default values when no predictions are submitted."""
        return {
            "bertscore_recall": {"task": self.task, "value": 0.0},
            "bertscore_weighted_recall": {
                "task": self.task,
                "value": 0.0,
            },
        }

    def compute_span_metrics(self, gold_spans: List[str], predicted_spans: List[str]):
        """Score predicted evidence spans by comparing with reference spans."""

        gold_spans = [s.strip() for s in gold_spans if s.strip()]
        predicted_spans = [s.strip() for s in predicted_spans if s.strip()]

        if not predicted_spans:
            return self.score_empty_predictions()

        # similar to CLPsych2024, we get timeline-level highlights;
        # assuming span order stays the same we should be able to analyze
        num_gold_spans_tokens = sum(
            len(_) - 2 for _ in self.tokenizer(gold_spans)["input_ids"]
        )
        num_pred_spans_tokens = sum(
            len(_) - 2 for _ in self.tokenizer(predicted_spans)["input_ids"]
        )
        if num_gold_spans_tokens < num_pred_spans_tokens:
            weight = num_gold_spans_tokens / num_pred_spans_tokens
        else:
            weight = 1.0

        curr_recalls, curr_recalls_weighted = [], []

        # for each expert evidence span, calculate maximum BERTScore
        for gold_span in gold_spans:
            _, _, F = self.scorer.score(
                predicted_spans, [gold_span] * len(predicted_spans)
            )
            score = F.max().item()
            curr_recalls.append(score)
            curr_recalls_weighted.append(score * weight)

        recall = np.array(curr_recalls)
        recall_weighted = np.array(curr_recalls_weighted)

        return {
            "bertscore_recall": {"task": self.task, "value": recall.mean()},
            "bertscore_weighted_recall": {
                "task": self.task,
                "value": recall_weighted.mean(),
            },
        }
