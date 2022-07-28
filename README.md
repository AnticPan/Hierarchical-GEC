# Hierarchical-GEC v2

## Installation

Clone this repository and enter.

Create a python 3.7 virtual environment and run the following command:
```bash
pip install -r requirements.txt
```

## Datasets and Trained Model

All datasets used in the paper can be found [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data). 
The M2 format file should be converted to tsv format file with source sentence and target sentences pairs per line.

Our trained model can be downloaded [here](https://drive.google.com/file/d/143sUJ7shfC4WknRBzfKwZrEampBYxz64/view?usp=sharing).

## Train Model

1. Download BERT or SpanBERT from [here](https://huggingface.co/models).
2. Prepare train and dev datasets.
3. Run the following command:
```bash
python train.py --bert_dir [BERT_DIR] \
                --train_file [TRAIN_FILE/DIR] \
                --valid_file [DEV_FILE/DIR] \
                --output_dir [OUTPUT_DIR] \
                --gpu 0 \
                --truncate 50 \
                --epoch 5 \
                --batch_size 128 \
                --lr 3e-5
```
4. The trained model is in [OUTPUT_DIR]/model/
5. Select the best model with `eval.sh`, [ERRANT 2.0.0](https://github.com/chrisjbryant/errant/tree/bea2019st) is used to evaluate the F0.5 score in development set and a Python3.6 environment is required for evaluation.



## Predict

### Choose Threshold

The default threshold is 0.5, you can find a better one by grid search in the development set.
Our trained model's threshold in CoNLL-2014 is 0.89 and in BEA-2019 is 0.72.

1. Set the `model_dir` and `valid_file` in `grid.sh`
2. Run `bash grid.sh` 

### Predict File

```bash
python predict.py --model_dir [TRAINED_MODEL_DIR] \
                  --output_dir [OUTPUT_DIR] \
                  --test_file [TEST_FILE] \
                  --discriminating_threshold [0.5] \
                  --batch_size 16 \
                  --gpu 0
```