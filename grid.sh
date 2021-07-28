#!/bin/bash
model_dir="[PATH OF TRAINED MODEL]"
valid_file="[PATH OF VALID FILE (.tsv)]" # absolute path
gpu=0

python predict.py --model_dir $model_dir --test_file $valid_file --output_dir /tmp --gpu $gpu -detecting -correcting
python grid.py --valid_file $valid_file --valid_pkl /tmp/$(basename $valid_file .tsv).output.pkl