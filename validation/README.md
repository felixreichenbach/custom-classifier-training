# Trained Model Validation

## Validation Data Export

```shell
# OK
viam dataset export --destination=./dataset --dataset-id=68b6cdd5ea9c776f67b19b00 --include-jsonl=true

# NOK
viam dataset export --destination=./dataset --dataset-id=68b6cdfd137cee2ab624cda4 --include-jsonl=true

```

## Process Images and Upload

1. Check aspect ratios

```shell
python image_aspect_ratio.py
```

2. Center crop images to 4:3

```shell
python center_crop_4_3.py
```

3. Update `tags`and `dataset id` in the `upload_cropped_images.py`
4. Update `.env`variables
5. Upload cropped images

```shell
python upload_cropped_images.py
```
