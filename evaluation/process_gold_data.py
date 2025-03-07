"""
Script to prepare gold standard annotations from raw exports.
Will produce file with empty/None fields if source files are un-annotated.
"""

import os
import json
from config import (
    DATA_DIR,
    DEV_PATHS,
    TEST_PATHS,
    DEV_ANNOTATED_FILENAME,
    TEST_ANNOTATED_FILENAME,
)
from nltk import sent_tokenize
import argparse
import logging

logger = logging.getLogger("process_gold_data")
logging.basicConfig(level=logging.INFO)


def process_annotated_data(data):
    """
    Process raw annotated timeline data into the format expected for evaluation.

    Args:
        data (dict): Raw annotated data loaded from the JSON, containing:
            - timeline_summary (str): Timeline-level self-state summary
            - posts (List[dict]): List of posts, each containing:
                - post (str): Post text
                - post_id (str): Post ID
                - Post Summary (str): Post-level self-state summary
                - evidence (Dict[str, List[str]]): Adaptive and maladaptive self-state evidence spans
                - Well-being (int): Wellbeing score

    Returns:
        dict: Extracted annotations:
            timeline_level:
                - summary (str): Timeline-level self-state summary
                - summary_sents (List[str]): Timeline-level self-state summary split into sentences
                - adaptive_spans (List[dict]): Adaptive evidence from all posts in the timeline,
                                               each a dict with text, element, and subcategory
                - maladaptive_spans (List[dict]): Maladaptive evidence from all posts in the timeline,
                                                  each a dict with text, element, and subcategory
                - sents (List[List[str]]): Sentences for each post in the timeline
                - post_ids (List[str]): Ordered post IDs in the timeline
            post_level:
                Dictionary mapping post_id to post information:
                - summary (str): Post-level self-state summary
                - summary_sents (List[str]): Post-level self-state summary split into sentences
                - wellbeing_score (float): Wellbeing score for the post
    """
    timeline_summary = data["timeline_summary"].strip()
    timeline_summary_sents = [
        s.strip() for s in sent_tokenize(timeline_summary) if s.strip()
    ]
    timeline_sents = []
    adaptive_spans = []
    maladaptive_spans = []
    # Python >= 3.6 keep insertion order for Dicts, but we explicitly include just in case
    post_ids = []

    post_level = dict()

    for post in data["posts"]:
        text = post.get("post", "")
        post_id = post["post_id"]
        summary = post.get("Post Summary", "")
        post_ids.append(post_id)
        if summary:
            summary = summary.strip()
            summary_sents = [s.strip() for s in sent_tokenize(summary) if s.strip()]
        else:
            summary_sents = []
        timeline_sents.append(
            [s.strip() for s in sent_tokenize(text) if s.strip()]
        )  # retain one list per post

        evidence = post.get("evidence")
        if evidence:
            adaptive_evidence = evidence.get("adaptive-state", dict())
            maladaptive_evidence = evidence.get("maladaptive-state", dict())
            for element, element_data in adaptive_evidence.items():
                adaptive_spans.append(
                    {
                        "text": element_data["highlighted_evidence"],
                        "element": element,
                        "subcategory": element_data["Category"],
                    }
                )
            for element, element_data in maladaptive_evidence.items():
                maladaptive_spans.append(
                    {
                        "text": element_data["highlighted_evidence"],
                        "element": element,
                        "subcategory": element_data["Category"],
                    }
                )

        post_level[post_id] = {
            "summary": summary,
            "summary_sents": summary_sents,
            "wellbeing_score": post.get("Well-being"),
        }

    return {
        "timeline_level": {
            "summary": timeline_summary,
            "summary_sents": timeline_summary_sents,
            "adaptive_spans": adaptive_spans,
            "maladaptive_spans": maladaptive_spans,
            "sents": timeline_sents,
            "post_ids": post_ids,
        },
        "post_level": post_level,
    }


def main(args):

    if args.test:
        gold_filename = TEST_ANNOTATED_FILENAME
        filepaths = TEST_PATHS
    else:
        gold_filename = DEV_ANNOTATED_FILENAME
        filepaths = DEV_PATHS

    gold_data_path = os.path.join(DATA_DIR, gold_filename)

    if os.path.exists(gold_data_path):
        logger.info(f"File exists at {gold_data_path}.")
        exit()

    gold_data = dict()
    for filepath in filepaths:
        with open(filepath, "r") as f:
            data = json.load(f)
        gold_data[data["timeline_id"]] = process_annotated_data(data)

    assert len(gold_data) == len(filepaths)

    with open(gold_data_path, "w") as f:
        json.dump(gold_data, f)

    logger.info(
        f"Saved processed annotations for {len(filepaths)} timelines to: {gold_data_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="If True, process test split. If False, process dev split (if specified in config.py)",
    )
    args = parser.parse_args()
    main(args)
