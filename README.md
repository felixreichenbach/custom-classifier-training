# Visual QA Training Script

Viam Custom Training Script Process: https://docs.viam.com/data-ai/train/train/

## Prerquisits

The libraries used are very sensitive to version changes!
The working combination for me was:

- Python 3.10
- tensorflow 2.14.1

## Setup Local Environment

```shell
python3.10 -m venv venv
source venv/bin/activate
```

## Export Dataset Locally

```shell
viam dataset export --destination=./dataset --dataset-id=687fca9c04f9d21febedb2dd --include-jsonl=true

viam dataset export --destination=<destination> --dataset-id=<dataset-id> --include-jsonl=true
```

## Train Locally

```shell
python3 ./model/training.py --dataset_file=./dataset/dataset.jsonl --model_type="single_label" \
    --model_output_directory=output \
    --labels='OK NOK'
```
