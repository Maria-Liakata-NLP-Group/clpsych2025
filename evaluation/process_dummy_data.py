"""
Script to create fake submission data for script validation.
"""

import argparse
import os
import json
import random
from nltk import sent_tokenize
from config import DEV_SUBMISSIONS_DIR, TEST_SUBMISSIONS_DIR, DEV_PATHS, TEST_PATHS
import logging

logger = logging.getLogger("process_dummy_data")
logging.basicConfig(level=logging.INFO)


def get_random_spans(text, num_spans=2, span_length=30):
    """Get random spans from text (may overlap)."""
    if len(text) < span_length:
        return [text]

    spans = []
    for _ in range(num_spans):
        start = random.randint(0, len(text) - span_length)
        spans.append(text[start : start + span_length])
    return spans


def get_random_sentences(text, num_sentences=2):
    """Get random sentences from text and join them."""
    sentences = sent_tokenize(text)
    if num_sentences >= len(sentences):
        return text
    return " ".join(random.sample(sentences, num_sentences))


def main(args):
    if args.test:
        dummy_path = os.path.join(TEST_SUBMISSIONS_DIR, "dummy_test.json")
        filepaths = TEST_PATHS
    else:
        dummy_path = os.path.join(DEV_SUBMISSIONS_DIR, "dummy_dev.json")
        filepaths = DEV_PATHS

    if os.path.exists(dummy_path):
        logger.info(f"File exists at {dummy_path}.")
        exit()

    dummy_submission_data = dict()

    for filepath in filepaths:
        with open(filepath, "r") as f:
            data = json.load(f)

        timeline_id = data["timeline_id"]
        timeline_concatenated = ""
        post_level = dict()

        for post in data["posts"]:
            text = post["post"]
            if len(text) > 100:
                adaptive_evidence = get_random_spans(
                    text,
                    num_spans=random.randint(1, 3),
                    span_length=random.randint(10, 20),
                )
                maladaptive_evidence = get_random_spans(
                    text,
                    num_spans=random.randint(1, 3),
                    span_length=random.randint(10, 20),
                )
            else:
                adaptive_evidence = get_random_spans(
                    text,
                    num_spans=random.randint(1, 2),
                    span_length=random.randint(1, max(len(text) - 30, 2)),
                )
                maladaptive_evidence = get_random_spans(
                    text,
                    num_spans=random.randint(1, 2),
                    span_length=random.randint(1, max(len(text) - 30, 2)),
                )

            post_level[post["post_id"]] = {
                "adaptive_evidence": adaptive_evidence,
                "maladaptive_evidence": maladaptive_evidence,
                "summary": get_random_sentences(
                    text, num_sentences=max(3, len(sent_tokenize(text)))
                ),
                "wellbeing_score": random.randint(1, 10),
            }
            timeline_concatenated += text + "\n"

        dummy_submission_data[timeline_id] = {
            "timeline_level": {
                "summary": get_random_sentences(timeline_concatenated, num_sentences=3)
            },
            "post_level": post_level,
        }

    assert len(dummy_submission_data) == len(filepaths)

    with open(dummy_path, "w") as f:
        json.dump(dummy_submission_data, f)
    logger.info(f"Saved dummy file with {len(filepaths)} timelines at: {dummy_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="If True, process test split. If False, process (internal) dev split.",
    )
    args = parser.parse_args()
    main(args)
