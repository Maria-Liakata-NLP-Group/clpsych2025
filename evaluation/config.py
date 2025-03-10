import os
import glob
from pathlib import Path

# Directory containing annotated data
TRAIN_DIR = (
    # replace with file paths on your system
    "/import/nlp/datasets/clpsych2025/train/"
)
TEST_DIR = (
    # replace with file paths on your system
    "/import/nlp/datasets/clpsych2025/test/"
)

# Randomly sampled timeline IDs from train for validation - replace to validate on new train timelines
DEV_TIMELINE_IDS = [
    "83997cd4e7",
    "46f4bb3ada",
    "0cac13e357",
    "5da839acb5",
    "6c9677b482",
]

# Base directories
EVAL_DIR = Path(os.path.abspath(__file__)).parent
DATA_DIR = os.path.join(EVAL_DIR, "data")
DEV_SUBMISSIONS_DIR = os.path.join(EVAL_DIR, "submissions_dev")
TEST_SUBMISSIONS_DIR = os.path.join(EVAL_DIR, "submissions")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# Path to timeline-post mapping
TIMELINE_POST_MAPPING_PATH = os.path.join(DATA_DIR, "timeline_id_to_post_id.json")

# Paths to individual JSON files
DEV_PATHS = [os.path.join(TRAIN_DIR, f"{tlid}.json") for tlid in DEV_TIMELINE_IDS]
TRAIN_PATHS = [
    p for p in glob.glob(TRAIN_DIR + "/*") if Path(p).stem not in DEV_TIMELINE_IDS
]
TEST_PATHS = [p for p in glob.glob(TEST_DIR + "/*")]
TEST_TIMELINE_IDS = [Path(p).stem for p in TEST_PATHS]

# Names of processed annotation files to be stored in data/
DEV_ANNOTATED_FILENAME = "dev.json"
TEST_ANNOTATED_FILENAME = "test.json"
