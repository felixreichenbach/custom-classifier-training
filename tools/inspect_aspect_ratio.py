import os
import json
from PIL import Image


def get_image_info(directory):
    image_info = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    width, height = img.size

                    def gcd(a, b):
                        while b:
                            a, b = b, a % b
                        return a

                    divisor = gcd(width, height)
                    aspect_ratio = (
                        f"{width // divisor}:{height // divisor}"
                        if height != 0
                        else None
                    )
                    image_info[filename] = {
                        "width": width,
                        "height": height,
                        "aspect_ratio": aspect_ratio,
                    }
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return image_info


if __name__ == "__main__":
    # Option 1: Use the current file's location (existing)
    # folder = os.path.join(os.path.dirname(__file__), "../dataset/data")

    # Option 2: Use the current working directory where the Python command was run from
    folder = os.path.join(os.getcwd(), "dataset_cropped_1_1/data")

    info = get_image_info(folder)
    # Summarize by aspect ratio
    summary = {}
    for details in info.values():
        ar = details["aspect_ratio"]
        if ar not in summary:
            summary[ar] = {
                "count": 0,
                "min_width": details["width"],
                "max_width": details["width"],
                "min_height": details["height"],
                "max_height": details["height"],
            }
        summary[ar]["count"] += 1
        summary[ar]["min_width"] = min(summary[ar]["min_width"], details["width"])
        summary[ar]["max_width"] = max(summary[ar]["max_width"], details["width"])
        summary[ar]["min_height"] = min(summary[ar]["min_height"], details["height"])
        summary[ar]["max_height"] = max(summary[ar]["max_height"], details["height"])
    print(json.dumps(summary, indent=2))
