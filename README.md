# Hierarchical-GEC

Code for our IEEE ICWS2021 short paper "Efficient Grammatical Error Correction with Hierarchical Error Detections and Correction".

## Installation

Clone this repository and enter.

Create a python 3.7 virtual environment and run the following command:
```bash
pip install -r requirements.txt
```

## Datasets and Trained Model

All datasets used in the paper can be found [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data). 
The M2 format file should be converted to tsv format file with source sentence and target sentences pairs per line, which can be done by using `utils/m2_to_tsv.py`.

Our trained model can be downloaded [here](https://drive.google.com/file/d/1KEWTuYnO3eM7QR9WJE1Xn5DVVzlD_V-T/view?usp=sharing).

## Train Model

1. Download BERT or SpanBERT from [here](https://huggingface.co/models).
2. Prepare train and dev datasets.
3. Run the following command:
```bash
python train.py --bert_dir [BERT_DIR] \
                --train_file [TRAIN_FILE/DIR] \
                --valid_file [DEV_FILE/DIR] \
                --output_dir [OUTPUT_DIR] \
                --gpus 0 \
                --truncate 50 \
                --epoch 3 \
                --batch_size 128 \
                --lr 3e-5
```
4. The trained model is in [OUTPUT_DIR]/model/epoch-[3]



## Predict

### Choose Threshold

The default threshold is 0.5, you can find a better one by grid search in the development set.

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

### Use gRPC

1. Start the gRPC server with command:
```bash
python grpc_server.py --model_dir [TRAINED_MODEL_DIR]
```
2. Call the api like `grpc_client.py`