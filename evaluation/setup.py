import json
from config import *

""""
Preparation for running scripts - download data, dir creation, mapping creation
"""

try:
    from nltk import sent_tokenize
except:
    import nltk

    nltk.download("punkt_tab")
    nltk.download("punkt")

for dir in [DATA_DIR, DEV_SUBMISSIONS_DIR, TEST_SUBMISSIONS_DIR, RESULTS_DIR]:
    Path(dir).mkdir(parents=True, exist_ok=True)

if not os.path.exists(TIMELINE_POST_MAPPING_PATH):
    timeline_id_to_post_id = dict()
    for paths in [TRAIN_PATHS, DEV_PATHS, TEST_PATHS]:
        for path in paths:
            with open(path, "r") as f:
                annotations = json.load(f)
            timeline_id = annotations["timeline_id"].strip()
            post_ids = [p["post_id"].strip() for p in annotations["posts"]]
            timeline_id_to_post_id[timeline_id] = post_ids
    with open(TIMELINE_POST_MAPPING_PATH, "w") as f:
        json.dump(timeline_id_to_post_id, f)
