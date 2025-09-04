import os
import shutil
from PIL import Image


def center_crop_to_aspect(img, target_ratio):
    width, height = img.size
    current_ratio = width / height
    if abs(current_ratio - target_ratio) < 1e-2:
        return img.copy()
    if current_ratio > target_ratio:
        # Crop width
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        box = (left, 0, right, height)
    else:
        # Crop height
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        box = (0, top, width, bottom)
    return img.crop(box)


def process_images(src_dir, dst_dir, target_ratio):
    src_dir = src_dir + "/data"
    dst_dir = dst_dir + "/data"
    os.makedirs(dst_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            try:
                with Image.open(src_path) as img:
                    cropped = center_crop_to_aspect(img, target_ratio)
                    cropped.save(dst_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    src_folder = os.path.join(os.getcwd(), "dataset_orig")
    dst_folder = os.path.join(os.getcwd(), "dataset_cropped_1_1")
    process_images(src_folder, dst_folder, 1.0)
    print(f"Cropped images saved to {dst_folder}")

    # Copy metadata folder and jsonl file
    shutil.copytree(
        src_folder + "/metadata",
        os.path.join(dst_folder, "metadata"),
        dirs_exist_ok=True,
    )
    shutil.copy(
        src_folder + "/dataset.jsonl", os.path.join(dst_folder, "dataset.jsonl")
    )
