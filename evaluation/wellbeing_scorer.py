from sklearn.metrics import mean_squared_error, f1_score
import numpy as np
from numpy.typing import ArrayLike


class WellbeingScorer:
    def __init__(self):
        self.task = "A.2"
        self.max_possible_error = 9  # wellbeing score range: [1, 10]
        # For optional binned analyses
        # Categories:
        # - Serious impairment to functioning (0): scores < 5
        # - Impaired functioning (1): scores 5-6
        # - Minimal impairment to functioning (2): scores > 6
        self.bins = {"serious": (1, 4), "impaired": (5, 6), "minimal": (7, 10)}

    def check_and_process_wellbeing_scores(
        self, y_trues: ArrayLike, y_preds: ArrayLike, do_penalize: bool = True
    ):
        """
        Validate and process wellbeing scores for evaluation.

        Args:
            y_trues: Gold wellbeing scores
            y_preds: Predicted wellbeing scores
            do_penalize: If True, penalize missing predictions

        Returns:
            (processed y_trues, processed y_preds, missing prediction mask)
        """
        if isinstance(y_trues, list):
            y_trues = np.array(y_trues)
        if isinstance(y_preds, list):
            y_preds = np.array(y_preds)

        # Skip indices where post is not annotated for wellbeing
        valid_gold_idx = y_trues != None
        y_trues = y_trues[valid_gold_idx]
        y_preds = y_preds[valid_gold_idx]

        # Handle missing predictions
        none_pred_idx = y_preds == None
        if not do_penalize:
            # Be lenient and skip indices where y_pred is None
            return (
                y_trues[~none_pred_idx].astype(int),
                y_preds[~none_pred_idx].astype(int),
                np.array([]),
            )
        return y_trues.astype(int), y_preds, none_pred_idx

    def bin_wellbeing_score(self, scores: ArrayLike):
        """Categorize wellbeing scores into severity levels."""
        scores = scores.copy()

        scores[
            (scores >= self.bins["serious"][0]) & (scores <= self.bins["serious"][1])
        ] = 0
        scores[
            (scores >= self.bins["impaired"][0]) & (scores <= self.bins["impaired"][1])
        ] = 1
        scores[
            (scores >= self.bins["minimal"][0]) & (scores <= self.bins["minimal"][1])
        ] = 2
        return scores

    def compute_mse(
        self,
        y_trues: ArrayLike,
        y_preds: ArrayLike,
        do_penalize: bool = True,
        do_binwise: bool = False,
        suffix: str = "",
    ):
        """
        Compute Mean Squared Error for wellbeing score predictions.

        Args:
            y_trues : Gold wellbeing scores
            y_preds : Predicted wellbeing scores
            do_penalize : If True, penalize incorrectly predicted Nones
            do_binwise: If True, will also compute MSE per wellbeing bin
            suffix: Optional marker to append to default metric name `mse'

        Returns:
            Dictionary containing MSE results
        """
        y_trues, y_preds, none_mask = self.check_and_process_wellbeing_scores(
            y_trues=y_trues, y_preds=y_preds, do_penalize=do_penalize
        )

        # Check if there is any empty prediction
        if none_mask.any():
            # 1. All empty
            if none_mask.all():
                max_error = self.max_possible_error
            else:
                # 2. Some empty;
                # Calculate maximum observed error from non-None predictions
                y_trues_subset = y_trues[~none_mask]
                y_preds_subset = y_preds[~none_mask].astype(int)
                max_error = np.max(np.abs(y_trues_subset - y_preds_subset))
            # Penalize None values
            y_preds[none_mask] = y_trues[none_mask] + max_error

        # 3. No empty prediction
        result = {
            f"mse_{suffix}" if suffix else "mse": {
                "task": "A.2",
                "value": mean_squared_error(y_trues, y_preds),
            }
        }
        if do_binwise:
            # Compute score for each category for optional analysis
            for bin_label, (bin_min, bin_max) in self.bins.items():
                bin_mask = (y_trues >= bin_min) & (y_trues <= bin_max)
                if not bin_mask.any():
                    continue
                result.update(
                    self.compute_mse(
                        y_trues=y_trues[bin_mask],
                        y_preds=y_preds[bin_mask],
                        do_penalize=do_penalize,
                        do_binwise=False,
                        suffix=bin_label,
                    )
                )
        return result
