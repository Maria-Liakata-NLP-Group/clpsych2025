"""
End-to-end evaluation script for the CLPsych 2025 Shared Task.
"""

import os
import argparse
import glob
from nltk import sent_tokenize
import json
import ast
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
import numpy as np

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

    for timeline_id, gold_datum in tqdm(gold_data.items()):

        curr_results = []

        predicted_spans_adaptive = []
        predicted_spans_maladaptive = []
        predicted_wellbeing_scores = []
        predicted_summary_sents = []

        # Exploratory: spans from both categories that preserve post structure
        predicted_spans = []

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

            adaptive_evidence = post_datum.get("adaptive_evidence", [])
            if isinstance(adaptive_evidence, str):
                try:
                    adaptive_evidence = ast.literal_eval(adaptive_evidence)
                    if not isinstance(adaptive_evidence, list):
                        adaptive_evidence = []
                except:
                    adaptive_evidence = []
            maladaptive_evidence = post_datum.get("maladaptive_evidence", [])
            if isinstance(maladaptive_evidence, str):
                try:
                    maladaptive_evidence = ast.literal_eval(maladaptive_evidence)
                    if not isinstance(maladaptive_evidence, list):
                        maladaptive_evidence = []
                except:
                    maladaptive_evidence = []

            predicted_spans_adaptive.extend(adaptive_evidence)
            predicted_spans_maladaptive.extend(maladaptive_evidence)
            # Exploratory: spans from both categories that preserve post structure
            predicted_spans.append(adaptive_evidence + maladaptive_evidence)

            wellbeing_score = post_datum.get("wellbeing_score")
            if isinstance(wellbeing_score, str):
                wellbeing_score = wellbeing_score.strip()
                if wellbeing_score and wellbeing_score.isnumeric():
                    wellbeing_score = float(wellbeing_score)
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
            # we only care about this for the present analysis
            curr_result_adaptive = curr_result_adaptive["bertscore_recall"]
            curr_result_maladaptive = curr_result_maladaptive["bertscore_recall"]

            curr_results.append(
                {
                    "timeline_id": timeline_id,
                    "post_id": None,
                    "task": "A.1-both",
                    "y": gold_spans_adaptive + gold_spans_maladaptive,
                    "yhat": predicted_spans_adaptive + predicted_spans_maladaptive,
                    "value": np.nanmean(
                        [
                            curr_result_adaptive["value"],
                            curr_result_maladaptive["value"],
                        ]
                    ),
                }
            )

            curr_results.append(
                {
                    "timeline_id": timeline_id,
                    "post_id": None,
                    "task": "A.1-adaptive",
                    "y": gold_spans_adaptive,
                    "yhat": predicted_spans_adaptive,
                    "value": curr_result_adaptive["value"],
                }
            )

            curr_results.append(
                {
                    "timeline_id": timeline_id,
                    "post_id": None,
                    "task": "A.1-maladaptive",
                    "y": gold_spans_maladaptive,
                    "yhat": predicted_spans_maladaptive,
                    "value": curr_result_maladaptive["value"],
                }
            )

        # Task A.2
        if do_A2:
            # we want the errors directly
            for post_index, pid in enumerate(post_ids):
                gold_score = gold_datum["post_level"][pid]["wellbeing_score"]
                predicted_score = predicted_wellbeing_scores[post_index]
                # we are less interested in abstaining from predictions
                if gold_score is None or predicted_score is None:
                    continue
                curr_results.append(
                    {
                        "timeline_id": timeline_id,
                        "post_id": pid,
                        "task": "A.2",
                        "y": gold_score,
                        "yhat": predicted_score,
                        "value": (gold_score - predicted_score) ** 2,  # squared error
                    }
                )

        if do_B or do_C:
            nli = NLIScorer()

        # Task B
        if do_B:
            for (curr_predicted_summary_sents, pid) in zip(
                predicted_summary_sents, post_ids
            ):
                curr_gold_summary_sents = gold_datum["post_level"][pid]["summary_sents"]
                # Evaluate only if there is a non-empty gold summary
                if curr_gold_summary_sents:
                    # Main metric: mean consistency with gold summary
                    # we skip exploratory metric
                    curr_result = nli.compute_post_nli_gold(
                        gold_sents=curr_gold_summary_sents,
                        predicted_sents=curr_predicted_summary_sents,
                    )["post_mean_consistency_gold"]["value"]

                    curr_results.append(
                        {
                            "timeline_id": timeline_id,
                            "post_id": pid,
                            "task": "B",
                            "y": ". ".join(curr_gold_summary_sents),
                            "yhat": ". ".join(curr_predicted_summary_sents),
                            "value": curr_result,
                        }
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
                curr_result = nli.compute_post_nli_gold(
                    gold_sents=gold_summary_sents_timeline,
                    predicted_sents=predicted_summary_sents_timeline,
                )["timeline_mean_consistency_gold"]["value"]
                # Main metric: mean consistency with gold summary
                curr_results.append(
                    {
                        "timeline_id": timeline_id,
                        "post_id": None,
                        "task": "C",
                        "y": ". ".join(gold_summary_sents_timeline),
                        "yhat": ". ".join(predicted_summary_sents_timeline),
                        "value": curr_result,
                    }
                )

        timeline_to_results[timeline_id] = curr_results

    return timeline_to_results


def main(args):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    submission_pattern = f"{args.team}*.json"

    if args.test:
        submission_data_paths = glob.glob(
            os.path.join(TEST_SUBMISSIONS_DIR, submission_pattern)
        )
        gold_filename = TEST_ANNOTATED_FILENAME
        evaluation_results_path = os.path.join(
            RESULTS_DIR, f"results_test_{timestamp}.csv"
        )
    else:
        submission_data_paths = glob.glob(
            os.path.join(DEV_SUBMISSIONS_DIR, submission_pattern)
        )
        gold_filename = DEV_ANNOTATED_FILENAME
        evaluation_results_path = os.path.join(
            RESULTS_DIR, f"results_dev_{timestamp}.csv"
        )

    if not submission_data_paths:
        logging.error(f"No submission files found in {submission_data_paths}")
        exit()

    with open(os.path.join(DATA_DIR, gold_filename), "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    do_A1, do_A2, do_B, do_C = get_active_tasks(args.tasks)

    results = defaultdict(list)
    for submission_data_path in tqdm(submission_data_paths):
        team_name, submission_id = parse_filename(submission_data_path)
        logger.info(f"Processing {Path(submission_data_path).name}")

        with open(submission_data_path, "r", encoding="utf-8") as f:
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
                    results["post_id"].append(metric_vals["post_id"])
                    results["y"].append(metric_vals["y"])
                    results["yhat"].append(metric_vals["yhat"])
                    results["team_name"].append(team_name)
                    results["submission_id"].append(submission_id)

    results_df = pd.DataFrame(results)
    results_df.to_csv(evaluation_results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="If True, run on test split. If False, run on dev split (assumes specified in config and processed).",
    )
    parser.add_argument(
        "--team",
        type=str,
        default="*",
        help="An optional pattern to evaluate only on selected files.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[],
        help="Tasks on which to evaluate submissions, out of A1, A2, B, C. If unspecified, evaluate all.",
    )
    args = parser.parse_args()
    main(args)
