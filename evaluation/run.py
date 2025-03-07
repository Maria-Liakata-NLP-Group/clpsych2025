"""
End-to-end evaluation script for the CLPsych 2025 Shared Task.
"""

import os
import argparse
import glob
from nltk import sent_tokenize
import json
from collections import defaultdict
from tqdm.auto import tqdm
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from span_scorer import SpanScorer
from wellbeing_scorer import WellbeingScorer
from nli_scorer import NLIScorer
from config import (
    DATA_DIR,
    DEV_SUBMISSIONS_DIR,
    TEST_SUBMISSIONS_DIR,
    RESULTS_DIR,
    DEV_ANNOTATED_FILENAME,
    TEST_ANNOTATED_FILENAME,
)

logger = logging.getLogger("run")
logging.basicConfig(level=logging.INFO)


def get_active_tasks(tasks):
    valid_tasks = ["A1", "A2", "B", "C"]

    invalid_tasks = set(tasks) - set(valid_tasks)
    if invalid_tasks:
        raise ValueError(
            f"Invalid tasks: {invalid_tasks}. Valid tasks are: {valid_tasks}"
        )

    if not tasks:
        # If no tasks specified, run all
        tasks = valid_tasks

    logging.info(f"Running evaluation on tasks: {str(tasks)}")
    return [valid_task in tasks for valid_task in valid_tasks]


def parse_filename(submission_data_path):
    # TODO parse and store team name, submissionID properly
    stem = Path(submission_data_path).stem
    eles = stem.split("_")
    if len(eles) == 2:
        team_name, submission_id = eles
    else:
        team_name = stem
        submission_id = stem
    return team_name, submission_id


def score_submission(
    submission_data, gold_data, do_A1=True, do_A2=True, do_B=True, do_C=True
):

    timeline_to_results = dict()

    # ss, ws, nli = load_scorers(do_A1=do_A1, do_A2=do_A2, do_B=do_B, do_C=do_C)

    for timeline_id, gold_datum in tqdm(gold_data.items()):

        curr_results = []

        predicted_spans_adaptive = []
        predicted_spans_maladaptive = []
        predicted_wellbeing_scores = []
        predicted_summary_sents = []

        # Get gold data
        post_ids = gold_datum["timeline_level"]["post_ids"]
        gold_spans_adaptive = [
            s["text"] for s in gold_datum["timeline_level"]["adaptive_spans"]
        ]
        gold_spans_maladaptive = [
            s["text"] for s in gold_datum["timeline_level"]["maladaptive_spans"]
        ]

        for post_id in post_ids:
            # Get prediction per post with type conversion & null handling
            post_datum = submission_data[timeline_id]["post_level"][post_id]
            predicted_spans_adaptive.extend(post_datum.get("adaptive_evidence", []))
            predicted_spans_maladaptive.extend(
                post_datum.get("maladaptive_evidence", [])
            )
            wellbeing_score = post_datum.get("wellbeing_score")
            if isinstance(wellbeing_score, str):
                wellbeing_score = wellbeing_score.strip()
                if wellbeing_score and wellbeing_score.isnumeric():
                    wellbeing_score = int(wellbeing_score)
                else:
                    wellbeing_score = None
            elif not (
                isinstance(wellbeing_score, int) or isinstance(wellbeing_score, float)
            ):
                wellbeing_score = None
            predicted_wellbeing_scores.append(wellbeing_score)
            post_summary = post_datum.get("summary", "")
            predicted_summary_sents.append(
                [s.strip() for s in sent_tokenize(post_summary) if s.strip()]
            )

        # Task A.1
        if do_A1:

            ss = SpanScorer()

            curr_result_adaptive = ss.compute_span_metrics(
                gold_spans=gold_spans_adaptive,
                predicted_spans=predicted_spans_adaptive,
            )
            curr_result_maladaptive = ss.compute_span_metrics(
                gold_spans=gold_spans_maladaptive,
                predicted_spans=predicted_spans_maladaptive,
            )
            # Main metric: store adaptive and maldaptive performance with equal weighting
            curr_results.append(curr_result_adaptive)
            curr_results.append(curr_result_maladaptive)

            # Store adaptive and maladaptive individually (treat as another metric)
            curr_results.append(
                {
                    metric_name + "_adaptive": v
                    for metric_name, v in curr_result_adaptive.items()
                }
            )
            curr_results.append(
                {
                    metric_name + "_maladaptive": v
                    for metric_name, v in curr_result_maladaptive.items()
                }
            )

        # Task A.2
        if do_A2:
            ws = WellbeingScorer()
            gold_wellbeing_scores = [
                gold_datum["post_level"][pid]["wellbeing_score"] for pid in post_ids
            ]
            # Main metric: MSE
            curr_results.append(
                ws.compute_mse(
                    y_trues=gold_wellbeing_scores,
                    y_preds=predicted_wellbeing_scores,
                    do_binwise=True,  # computes MSE per bin for optional analysis
                )
            )

        if do_B or do_C:
            nli = NLIScorer()

        # Task B
        if do_B:
            gold_summary_sents = [
                gold_datum["post_level"][pid]["summary_sents"] for pid in post_ids
            ]
            for (
                curr_gold_summary_sents,
                curr_predicted_summary_sents,
            ) in zip(gold_summary_sents, predicted_summary_sents):

                # Evaluate only if there is a non-empty gold summary
                if curr_gold_summary_sents:
                    # Main metric: mean consistency with gold summary
                    curr_results.append(
                        nli.compute_post_nli_gold(
                            gold_sents=curr_gold_summary_sents,
                            predicted_sents=curr_predicted_summary_sents,
                        )
                    )

        # Task C
        if do_C:
            timeline_summary = submission_data[timeline_id]["timeline_level"]["summary"]
            predicted_summary_sents_timeline = []
            if isinstance(timeline_summary, str):
                predicted_summary_sents_timeline.extend(
                    [s.strip() for s in sent_tokenize(timeline_summary) if s.strip()]
                )

            gold_summary_sents_timeline = gold_datum["timeline_level"]["summary_sents"]

            # Evaluate only if there is a non-empty gold summary
            if gold_summary_sents_timeline:
                # Main metric: mean consistency with gold summary
                curr_results.append(
                    nli.compute_timeline_nli_gold(
                        gold_sents=gold_summary_sents_timeline,
                        predicted_sents=predicted_summary_sents_timeline,
                    )
                )

        timeline_to_results[timeline_id] = curr_results

    return timeline_to_results


def main(args):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.test:
        submission_data_paths = glob.glob(os.path.join(TEST_SUBMISSIONS_DIR, "*.json"))
        gold_filename = TEST_ANNOTATED_FILENAME
        evaluation_results_path = os.path.join(
            RESULTS_DIR, f"results_test_{timestamp}.csv"
        )
    else:
        submission_data_paths = glob.glob(os.path.join(DEV_SUBMISSIONS_DIR, "*.json"))
        gold_filename = DEV_ANNOTATED_FILENAME
        evaluation_results_path = os.path.join(
            RESULTS_DIR, f"results_dev_{timestamp}.csv"
        )

    if not submission_data_paths:
        logging.error(f"No submission files found in {submission_data_paths}")
        exit()

    with open(os.path.join(DATA_DIR, gold_filename), "r") as f:
        gold_data = json.load(f)

    do_A1, do_A2, do_B, do_C = get_active_tasks(args.tasks)

    results = defaultdict(list)
    for submission_data_path in tqdm(submission_data_paths):
        team_name, submission_id = parse_filename(submission_data_path)
        logger.info(f"Processing {Path(submission_data_path).name}")

        with open(submission_data_path, "r") as f:
            submission_data = json.load(f)

        for timeline_id, timeline_results in score_submission(
            submission_data=submission_data,
            gold_data=gold_data,
            do_A1=do_A1,
            do_A2=do_A2,
            do_B=do_B,
            do_C=do_C,
        ).items():
            for curr_result in timeline_results:
                for metric_name, metric_vals in curr_result.items():
                    results["timeline_id"].append(timeline_id)
                    results["metric"].append(metric_name)
                    results["task"].append(metric_vals["task"])
                    results["value"].append(metric_vals["value"])
                    results["team_name"].append(team_name)
                    results["submission_id"].append(submission_id)

    results_df = pd.DataFrame(results)
    results_df.to_csv(evaluation_results_path)
    print(
        results_df.groupby(
            ["team_name", "submission_id", "task", "metric"]
        ).value.mean()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="If True, run on test split. If False, run on dev split (assumes specified in config and processed).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[],
        help="Tasks on which to evaluate submissions, out of A1, A2, B, C. If unspecified, evaluate all.",
    )
    args = parser.parse_args()
    main(args)
