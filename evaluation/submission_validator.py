#!/usr/bin/env python3
"""
Submission validator for CLPsych 2025 Shared Task:
checks that JSON file matches required format for evaluation, 
including structure, field names, types, and timeline-post mappings.

Requires there to be a mapping files created in data/ (see setup.sh).
"""
import argparse
import json
import logging
import sys
from config import TIMELINE_POST_MAPPING_PATH, DEV_TIMELINE_IDS, TEST_TIMELINE_IDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class Validator:
    def __init__(self, args):
        self.valid = True
        with open(TIMELINE_POST_MAPPING_PATH, "r") as f:
            if args.dev:
                # Check that files exist in expected locations;
                # otherwise it will affect timeline-post-mapping checks
                if not DEV_TIMELINE_IDS:
                    raise FileNotFoundError(
                        "Unable to load dev timelines. Check config.py to see if paths are set correctly."
                    )
                self.timeline_id_to_post_ids = {
                    tlid: pids
                    for (tlid, pids) in json.load(f).items()
                    if tlid in DEV_TIMELINE_IDS
                }
            else:
                if not TEST_TIMELINE_IDS:
                    raise FileNotFoundError(
                        "Unable to load test timelines. Check config.py to see if paths are set correctly."
                    )
                self.timeline_id_to_post_ids = {
                    tlid: pids
                    for (tlid, pids) in json.load(f).items()
                    if tlid in TEST_TIMELINE_IDS
                }
        self.timelines_with_issues = set()
        self.posts_with_issues = set()  # Will store (timeline_id, post_id) tuples

    ######################### Helpers #########################

    def log_timeline_issue(self, timeline_id):
        self.timelines_with_issues.add(timeline_id)
        self.valid = False

    def log_post_issue(self, timeline_id, post_id):
        self.posts_with_issues.add((timeline_id, post_id))
        self.log_timeline_issue(timeline_id)

    def check_type(self, value, expected_type, context, timeline_id=None, post_id=None):
        if not isinstance(value, expected_type):
            type_name = expected_type.__name__
            logger.error(f"{context}: Expected {type_name}, got {type(value).__name__}")

            if post_id and timeline_id:
                self.log_post_issue(timeline_id, post_id)
            elif timeline_id:
                self.log_timeline_issue(timeline_id)
            else:
                self.valid = False

            return False
        return True

    def check_required_fields(
        self, data, required_fields, context, timeline_id=None, post_id=None
    ):
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            logger.error(f"{context}: Missing required fields: {missing_fields}")

            if post_id and timeline_id:
                self.log_post_issue(timeline_id, post_id)
            elif timeline_id:
                self.log_timeline_issue(timeline_id)
            else:
                self.valid = False

            return False
        return True

    ######################### Validation #########################
    # A file contains timeline_id to timeline dict mappings
    # where each timeline dict contains timeline_level and post_level fields
    # and post_level contains post_id to post dict mappings
    def validate_file(self, file_path):
        """Validate the JSON file at the specified path."""
        logger.info(f"Validating JSON file: {file_path}")

        # Check can read file properly
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    self.valid = False
                    return
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            self.valid = False
            return
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            self.valid = False
            return

        # Check if the root is a non-empty dictionary
        if not data:
            logger.error("JSON is empty (no timelines)")
            self.valid = False
            return

        if not self.check_type(data, dict, "Submission file"):
            return

        # Check for missing timelines in the submission
        try:
            expected_timelines = set(self.timeline_id_to_post_ids.keys())
            actual_timelines = set(data.keys())

            missing_timelines = expected_timelines - actual_timelines
            if missing_timelines:
                logger.error(f"Missing expected timelines: {missing_timelines}")
                self.valid = False
                for missing_timeline in missing_timelines:
                    self.log_timeline_issue(timeline_id=missing_timeline)

            unexpected_timelines = actual_timelines - expected_timelines
            if unexpected_timelines:
                logger.warning(
                    f"Found unexpected timelines: {unexpected_timelines}. They will be ignored during evaluation."
                )
        except Exception as e:
            logger.error(f"Error checking timeline mapping: {e}")
            self.valid = False

        # Validate each timeline
        for timeline_id, timeline_data in data.items():
            try:
                logger.info(f"Validating timeline: {timeline_id}")
                self.validate_timeline_dict(timeline_data, timeline_id)
            except Exception as e:
                logger.error(f"Error validating timeline {timeline_id}: {e}")
                self.log_timeline_issue(timeline_id)

        # Log validation result with issue counts
        if self.valid:
            logger.info("JSON file is valid.")
        else:
            timeline_issue_count = len(self.timelines_with_issues)
            post_issue_count = len(self.posts_with_issues)

            logger.error(f"JSON file is invalid.")
            logger.error(f"Found {timeline_issue_count} timelines with issues")
            logger.error(f"Found {post_issue_count} unique posts with issues")

    def validate_timeline_dict(self, timeline_data, timeline_id):
        """Validate a timeline's structure and fields."""
        context = f"Timeline {timeline_id}"

        # Check if timeline_data is a non-empty dictionary
        if not self.check_type(timeline_data, dict, context, timeline_id):
            return
        if not timeline_data:
            self.valid = False
            self.log_timeline_issue(timeline_id)
            return

        # Check required fields
        required_fields = {"timeline_level", "post_level"}
        self.check_required_fields(timeline_data, required_fields, context, timeline_id)

        # Validate timeline_level
        try:
            if "timeline_level" in timeline_data:
                self.validate_timeline_level(
                    timeline_data["timeline_level"], timeline_id
                )
        except Exception as e:
            logger.error(f"{context}: Error validating timeline_level: {e}")
            self.log_timeline_issue(timeline_id)

        # Validate post_level
        try:
            if "post_level" in timeline_data:
                self.validate_post_level(timeline_data["post_level"], timeline_id)
        except Exception as e:
            logger.error(f"{context}: Error validating post_level: {e}")
            self.log_timeline_issue(timeline_id)

    def validate_timeline_level(self, timeline_level, timeline_id):
        """Validate the timeline_level structure and fields."""
        context = f"Timeline {timeline_id}: timeline_level"

        # Check if is non-empty dictionary - if's not, we can't check individual fields, so log and continue
        if not self.check_type(timeline_level, dict, context, timeline_id):
            return
        if not timeline_level:
            logger.error(f"{context} is empty")
            self.log_timeline_issue(timeline_id)
            self.valid = False
            return

        # Check required fields
        required_fields = {"summary"}
        if not self.check_required_fields(
            timeline_level, required_fields, context, timeline_id
        ):
            return

        # Check summary is a string
        self.check_type(
            timeline_level["summary"], str, f"{context} summary", timeline_id
        )

        # Warn if summary is empty (only if it's a string)
        if (
            isinstance(timeline_level["summary"], str)
            and not timeline_level["summary"].strip()
        ):
            logger.warning(f"{context} summary is empty")

    def validate_post_level(self, post_level, timeline_id):
        """Validate the post_level structure and all posts within it."""
        context = f"Timeline {timeline_id}: post_level"

        # Check if post_level is a non-empty dictionary - if not, log and continue
        if not self.check_type(post_level, dict, context, timeline_id):
            return

        if not post_level:
            logger.error(f"{context} contains no posts")
            self.log_timeline_issue(timeline_id)
            self.valid = False
            return

        # Get the expected post_ids for this timeline
        try:
            expected_post_ids = set(self.timeline_id_to_post_ids.get(timeline_id, []))
            actual_post_ids = set(post_level.keys())

            # Check for missing expected posts
            missing_posts = expected_post_ids - actual_post_ids
            if missing_posts:
                logger.error(f"{context}: Missing expected posts: {missing_posts}")
                for missing_post in missing_posts:
                    self.log_post_issue(timeline_id=timeline_id, post_id=missing_post)

            # Warn if contains unexpected posts - they will be skipped during evaluation
            unexpected_posts = actual_post_ids - expected_post_ids
            if unexpected_posts:
                logger.warning(
                    f"{context}: Found unexpected posts: {unexpected_posts}.  They will be ignored during evaluation."
                )

        except Exception as e:
            logger.error(f"{context}: Error checking post mapping: {e}")
            self.log_timeline_issue(timeline_id)

        # Validate each post
        for post_id, post_data in post_level.items():
            try:
                self.validate_post_dict(post_data, timeline_id, post_id)
            except Exception as e:
                logger.error(
                    f"Error validating post {post_id} in timeline {timeline_id}: {e}"
                )
                self.log_post_issue(timeline_id, post_id)

    def validate_post_dict(self, post_data, timeline_id, post_id):
        """Validate a single post's prediction have expected structure and fields."""
        context = f"Timeline {timeline_id}, Post {post_id}"

        # Check if post_data is a non-empty dictionary; if not, we can't check fields, so we log and continue
        if not self.check_type(post_data, dict, context, timeline_id, post_id):
            return
        if not post_data:
            self.valid = False
            self.log_post_issue(timeline_id, post_id)
            return

        # Check required fields
        required_fields = {
            "adaptive_evidence",
            "maladaptive_evidence",
            "summary",
            "wellbeing_score",
        }
        self.check_required_fields(
            post_data, required_fields, context, timeline_id, post_id
        )

        # Validate adaptive_evidence
        if "adaptive_evidence" in post_data:
            if self.check_type(
                post_data["adaptive_evidence"],
                list,
                f"{context}: adaptive_evidence",
                timeline_id,
                post_id,
            ):
                try:
                    for i, evidence in enumerate(post_data["adaptive_evidence"]):
                        self.check_type(
                            evidence,
                            str,
                            f"{context}: adaptive_evidence[{i}]",
                            timeline_id,
                            post_id,
                        )
                except Exception as e:
                    logger.error(f"{context}: Error validating adaptive_evidence: {e}")
                    self.log_post_issue(timeline_id, post_id)

        # Validate maladaptive_evidence
        if "maladaptive_evidence" in post_data:
            if self.check_type(
                post_data["maladaptive_evidence"],
                list,
                f"{context}: maladaptive_evidence",
                timeline_id,
                post_id,
            ):
                try:
                    for i, evidence in enumerate(post_data["maladaptive_evidence"]):
                        self.check_type(
                            evidence,
                            str,
                            f"{context}: maladaptive_evidence[{i}]",
                            timeline_id,
                            post_id,
                        )
                except Exception as e:
                    logger.error(
                        f"{context}: Error validating maladaptive_evidence: {e}"
                    )
                    self.log_post_issue(timeline_id, post_id)

        # Validate summary
        # If no summary is predicted, it should be an empty str
        if "summary" in post_data:
            self.check_type(
                post_data["summary"], str, f"{context}: summary", timeline_id, post_id
            )

        # Validate wellbeing_score
        if "wellbeing_score" in post_data:
            score = post_data["wellbeing_score"]
            # If no score is predicted, it should be None
            if score is None:
                return
            if isinstance(score, float):
                logger.warning(
                    f"{context}: Well-being score is float. It will be truncated to int automatically during evaluation."
                )
            elif not self.check_type(
                score,
                int,
                f"{context}: wellbeing_score",
                timeline_id,
                post_id,
            ):
                return
            # Validate score range
            if score < 1 or score > 10:
                logger.error(
                    f"{context}: Well-being score must be between 1 and 10 (inclusive), got {score}."
                )
                self.log_post_issue(timeline_id, post_id)


def main():
    parser = argparse.ArgumentParser(
        description="Validate a JSON submission file for the shared task"
    )
    parser.add_argument(
        "-f", "--file_path", required=True, help="Path to the JSON file to validate"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="If True, checks submission on dev split",
    )
    args = parser.parse_args()

    validator = Validator(args)
    validator.validate_file(args.file_path)

    sys.exit(0 if validator.valid else 1)


if __name__ == "__main__":
    main()
