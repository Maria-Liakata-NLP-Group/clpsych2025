import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List


# Task B, C
class NLIScorer:
    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)

    def _compute_nli_scores(self, premise: str, hypothesis: str):
        input = self.tokenizer(
            premise, hypothesis, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input["input_ids"].to(self.device))
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
        return prediction

    def compute_nli_scores(
        self,
        source_sents: List[str],
        predicted_sents: List[str],
        task: str,
        prefix: str,
        source_name: str,
    ):
        """
        Args:
            source_sents: List of source sentences to compare against
            predicted_sents: List of predicted sentences
            task: Task ID ('B' or 'C')
            prefix: Prefix for metric names ('post' or 'timeline')
            source_name: Name of the source ('gold' or 'post' or 'timeline')

        Returns:
            Dictionary with computed NLI metrics
        """
        if predicted_sents:
            entail_scores, contradict_scores = [], []
            for source_sent in source_sents:
                for predicted_sent in predicted_sents:
                    scores = self._compute_nli_scores(source_sent, predicted_sent)
                    entail_scores.append(scores["entailment"])
                    contradict_scores.append(scores["contradiction"])
            entail_scores, contradict_scores = np.array(entail_scores), np.array(
                contradict_scores
            )
            mean_consistency = 1 - contradict_scores.mean()
            max_entailment = entail_scores.max()
            max_contradiction = contradict_scores.max()
        else:
            mean_consistency = 0.0
            max_entailment = 0.0
            max_contradiction = 1.0

        return {
            f"{prefix}_mean_consistency_{source_name}": {
                "value": mean_consistency,
                "task": task,
            },
            f"{prefix}_max_entailment_{source_name}": {
                "value": max_entailment,
                "task": task,
            },
            f"{prefix}_max_contradiction_{source_name}": {
                "value": max_contradiction,
                "task": task,
            },
        }

    def compute_post_nli_gold(self, gold_sents: List[str], predicted_sents: List[str]):
        """Compute NLI scores for post-level summary against gold summary."""
        return self.compute_nli_scores(
            source_sents=gold_sents,
            predicted_sents=predicted_sents,
            task="B",
            prefix="post",
            source_name="gold",
        )

    def compute_timeline_nli_gold(
        self, gold_sents: List[str], predicted_sents: List[str]
    ):
        """Compute NLI scores for timeline-level summary against timeline summary."""
        return self.compute_nli_scores(
            source_sents=gold_sents,
            predicted_sents=predicted_sents,
            task="C",
            prefix="timeline",
            source_name="gold",
        )

    def compute_summary_nli_evidence(
        self, evidence_spans: List[str], summary_sents: List[str]
    ):
        """Helper for optional exploratory metric - evidence appropriateness"""
        return self.compute_nli_scores(
            source_sents=evidence_spans,
            predicted_sents=summary_sents,
            task="B",
            prefix="post",
            source_name="evidence",
        )
