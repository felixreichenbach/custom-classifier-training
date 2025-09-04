import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser(description="Split dataset into train and test sets.")
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to the dataset directory",
)
args = parser.parse_args()

src_dir = os.path.join(args.dataset, "data")
jsonl_src = os.path.join(args.dataset, "dataset.jsonl")


# src_dir = "./dataset_cropped_1_1/data"
# jsonl_src = "./dataset_cropped_1_1/dataset.jsonl"
train_dir = "./dataset_train"
test_dir = "./dataset_test"
split_ratio = 0.8

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)

# List all image files
image_files = [
    f
    for f in os.listdir(src_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
]
random.shuffle(image_files)

split_index = int(len(image_files) * split_ratio)
train_files = image_files[:split_index]
test_files = image_files[split_index:]

for fname in train_files:
    shutil.copy(os.path.join(src_dir, fname), os.path.join(train_dir + "/data", fname))

for fname in test_files:
    shutil.copy(os.path.join(src_dir, fname), os.path.join(test_dir + "/data", fname))

# Copy metadata folder and jsonl file to both train and test folders
for target_dir in [train_dir, test_dir]:
    shutil.copy(jsonl_src, os.path.join(target_dir, "dataset.jsonl"))

print(f"Train images: {len(train_files)}, Test images: {len(test_files)}")
print(f"Train folder: {train_dir}")
print(f"Test folder: {test_dir}")
