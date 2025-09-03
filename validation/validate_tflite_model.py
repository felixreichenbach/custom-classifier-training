import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ExifTags
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def run_inference(
    interpreter: tf.lite.Interpreter, input_details, output_details, img_array
):
    # img_array = img_array.astype(np.uint8)
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


def load_labels(model_path):
    labels_txt = os.path.join(os.path.dirname(model_path), "labels.txt")
    if os.path.exists(labels_txt):
        with open(labels_txt, "r") as f:
            labels = [line.strip() for line in f if line.strip()]
    else:
        labels = []
    print(f"Loaded labels: {labels}")
    return labels


def load_images(image_dir):
    print(f"Loading images from: {image_dir}")
    images_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
    ]
    return images_files


def preprocess_image(image_path, input_shape):
    img = Image.open(image_path).convert("RGB")
    exif = img.getexif()
    if exif:
        for tag, value in exif.items():
            if ExifTags.TAGS.get(tag) == "Orientation":
                print(f"Image orientation: {value}")
    else:
        print("No EXIF orientation found.")
    img = img.resize((input_shape[1], input_shape[2]))
    # Activate to visually check images after preprocessing
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
    img_array = np.array(img)
    if np.any(img_array < 0) or np.any(img_array > 255):
        raise ValueError("Image pixel values are out of uint8 range (0-255).")
    img_array = img_array.astype(np.uint8)
    # If img_array originally has shape (height, width, channels), after this operation, its shape becomes (1, height, width, channels).
    # Many machine learning models (especially TensorFlow Lite models) expect input data to have a batch dimension, even if you’re only passing one image.
    # Adding this dimension makes the array compatible with model input requirements.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def validate_model(model_path, dataset, output_json):
    result = []

    # Load TFLite model
    interpreter, input_details, output_details = load_tflite_model(model_path)
    input_shape = input_details[0]["shape"]

    image_dir = os.path.join(dataset, "data_cropped_4_3")
    image_files = load_images(image_dir)

    # Load labels from model/labels.txt
    labels = load_labels(model_path)

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        prep_image = preprocess_image(image_path, input_shape)
        output_data = run_inference(
            interpreter, input_details, output_details, prep_image
        )
        # Map output to label
        pred_class = int(np.argmax(output_data))
        label = labels[pred_class] if pred_class < len(labels) else None
        print(
            f"{filename}: {output_data} -> predicted class: {pred_class}, label: {label}"
        )
        result.append(
            {
                "filename": filename,
                "output": output_data.tolist(),
                "predicted_class": pred_class,
                "label": label,
            }
        )

    # Save results to output_json
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate TFLite model on images.")
    parser.add_argument("--model", required=True, help="Path to model.tflite")
    parser.add_argument("--dataset", required=True, help="Directory of images")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    validate_model(args.model, args.dataset, args.output)
