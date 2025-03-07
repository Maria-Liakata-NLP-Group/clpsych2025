#!/bin/bash

# Usage: ./setup.sh [--make-dummy]

set -e

# Create the relevant directories, timeline-to-post mappings for file validation, 
# and nltk data to run its tokenizer.
python setup.py

#########################
# As described in the README, this script will focus on dev validation.
#########################

# Extract annotations and put into a evaluation-ready format
python process_gold_data.py
# python process_gold_data.py --test

# Generate fake submission file for testing (optional)
if [ "$1" == "--make-dummy" ]; then
    python process_dummy_data.py
    # python process_dummy_data.py --test
fi

echo "Setup complete."