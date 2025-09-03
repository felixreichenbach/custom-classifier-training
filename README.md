# Visual QA Training Script

A single label classifier tflite model training script customizable to suite your needs.

Basically follows the instructions here:
Viam Custom Training Script Process: https://docs.viam.com/data-ai/train/train/

And uses this as example:
Script Example: https://github.com/viam-modules/classification-tflite

## Local Development Setup

### Prerequisits

The libraries used are very sensitive to version changes!
The working combination for me was:

- Python 3.10
- tensorflow 2.14.1

### Setup Local Environment

Create virtual environment:

```shell
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset Locally

You can download / export a data set in the Viam platform to a local folder using the following command:

```shell
viam dataset export --destination=<LOCAL-FOLDER> --dataset-id=<DATASET-ID> --include-jsonl=true
```

## Train Locally

To test the training script on your local system, use the following command:

```shell
python3 ./model/training.py \
    --dataset_file=<LOCAL-FOLDER>/dataset.jsonl \
    --model_type="single_label" \
    --model_output_directory=<LOCAL-FOLDER> \
    --labels='Label1 Label2' \
    --num_epochs=1
```

## Submit Cloud Training Job

Use this command to trigger a training process witht a training script already uploaded to the Viam registry:

```shell
viam train submit custom with-upload \
    --dataset-id=<DATASET-ID> \
    --model-org-id=<ORG-ID> \
    --model-name=classification \
    --model-type=<DESIRED_TYPE> \
    --framework=tflite \
    --path=<REPO-TAR-PATH> \
    --script-name=classification_script \
    --args=num_epochs=3,labels="'Label1 Label2'"
```

## Enable Github Actions

The repository contains two Github workflows, `pull_request.yaml` which will test your current script and `main.yaml` which will deploy the script into the Viam registry.

Both scripts require Github secrects as follows. You can add them via your repository -> settings -> secrets and variables.

```
VIAM_API_KEY_ID: < YOUR API KEY ID >
VIAM_API_KEY: < YOUR API KEY >
VIAM_ORG_ID: < YOUR ORGANIZATION ID >

# pull_request.yaml additionally requires a test data set
VIAM_DATASET_ID: < YOUR TEST DATASET ID >
```
