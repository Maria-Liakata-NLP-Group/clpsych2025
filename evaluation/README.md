# About

This repository contains scripts to run evaluation on CLPsych 2025 shared task submissions.

Since the annotated test data will not be released during the shared task, the example below shows how evaluation can be run on **validation timelines** randomly sampled from the train split.

# Setup
Install required packages in a new environment.
```
conda create -n clpsych_env python=3.9
conda activate clpsych_env
pip install -r requirements.txt
```
In `config.py`, set path to directories containing train and test data. Additionally, specify the timeline IDs of timelines you'd like to use for validation in `DEV_TIMELINE_IDS`. 

Then, set up the files. The `--make-dummy` flag is optional, and when selected it will create a fake submission file with random predictions, which can be used as naive baseline and for debugging.
```
bash setup.sh --make-dummy
```

# Usage
## Submission File 
Place your submission JSONs under `submission_dev/`, created during setup.

Check that the file is properly formatted using the file validator helper:
```
python submission_validator.py -f {path_to_your_json_submission} 
```

If the file content and structure is as expected, the output will look like this:
![Screenshot 2025-03-07 at 15 29 17](https://github.com/user-attachments/assets/d5396c35-6b9b-4512-9abe-9279810f507c)

And if there are errors, they will be displayed, e.g.:
![Screenshot 2025-03-07 at 15 29 41](https://github.com/user-attachments/assets/48ddfa25-ba8a-4dc4-867a-79115d0132c9)

![Screenshot 2025-03-07 at 15 30 01](https://github.com/user-attachments/assets/03ff29bb-72f2-4d3a-83ba-ef4a97edf05a)


## Evaluation
To evaluate all files in the submission directory on all tasks:
```
python evaluation/run.py
```
Or for task-speciic evaluation:
```
python run.py --test --tasks {your tasks}
```
for example:
```
python run.py --test --tasks A1 B
```
Results will be displayed on the terminal and saved as a CSV under `results/`.
