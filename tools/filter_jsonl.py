import os
import json

datasets = ["dataset_train", "dataset_test"]

for dataset in datasets:
    input_jsonl = dataset + "/dataset.jsonl"
    output_jsonl = dataset + "/dataset_filtered.jsonl"

    print(f"Filtering dataset: {input_jsonl}")
    print(f"Filtered dataset will be written to: {output_jsonl}")

    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def resolve_path(image_path):
        # If image_path is absolute, use as is; else, resolve relative to workspace
        if os.path.isabs(image_path):
            return image_path
        return os.path.join(workspace_dir, image_path)

    with open(input_jsonl, "r") as fin, open(output_jsonl, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            image_path = entry.get("image_path")
            image_path = image_path.replace("dataset_orig", dataset)
            print(image_path)
            resolved_path = resolve_path(image_path) if image_path else None
            if image_path and os.path.exists(resolved_path):
                fout.write(json.dumps(entry) + "\n")
            else:
                print(f"Image not found: {resolved_path}")
    print(f"Filtered dataset written to {output_jsonl}")
