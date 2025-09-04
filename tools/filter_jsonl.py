import os
import json

input_jsonl = "dataset_train/dataset.jsonl"
output_jsonl = "dataset_train/dataset_filtered.jsonl"

with open(input_jsonl, "r") as fin, open(output_jsonl, "w") as fout:
    for line in fin:
        entry = json.loads(line)
        image_path = entry.get("image_path")
        if image_path and os.path.exists(image_path):
            fout.write(json.dumps(entry) + "\n")

print(f"Filtered dataset written to {output_jsonl}")
