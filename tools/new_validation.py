import argparse
import json
import os
import sys
import typing as ty
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from collections import defaultdict
import matplotlib.pyplot as plt
import csv

labels_filename = "labels.txt"
metrics_filename = "validation_metrics.json"

ROUNDING_DIGITS = 5


def save_prediction_scores(filenames, y_true, y_probs, output_dir):
    """Save a CSV with filename, true label, and predicted probability."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "prediction_scores.csv")
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "true_label", "predicted_prob"])
        for fname, t, p in zip(filenames, y_true, y_probs):
            writer.writerow([fname, t, p])
    print(f"Prediction scores saved to {out_path}")


def plot_f1_vs_threshold(y_true, y_probs, output_dir):
    """Plot F1 score vs. threshold for binary classification."""
    thresholds = np.arange(0, 1.01, 0.01)
    f1s = [f1_score(y_true, y_probs > t) for t in thresholds]
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    plt.figure()
    plt.plot(thresholds, f1s, label="F1 Score")
    plt.axvline(
        best_threshold,
        color="r",
        linestyle="--",
        label=f"Best Threshold: {best_threshold:.2f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Threshold")
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "f1_vs_threshold.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"F1 vs. threshold plot saved to {plot_path}")


def parse_args(args):
    """
    Parses command-line arguments for the validation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", dest="data_json", type=str, required=True)
    parser.add_argument("--model_directory", dest="model_dir", type=str, required=True)
    parser.add_argument(
        "--output_directory", dest="output_dir", type=str, required=True
    )
    parsed_args = parser.parse_args(args)
    return parsed_args.data_json, parsed_args.model_dir, parsed_args.output_dir


def load_labels(model_dir: str) -> ty.List[str]:
    """
    Loads labels from the labels.txt file.
    """
    labels_path = os.path.join(model_dir, labels_filename)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


def parse_filenames_and_labels_from_json(
    filename: str, all_labels: ty.List[str]
) -> ty.Tuple[ty.List[str], ty.List[str]]:
    """
    Loads and parses JSON file to return image filenames and corresponding labels.
    This function must be identical to the one in the training script to ensure
    data consistency.
    """
    image_filenames = []
    image_labels = []
    with open(filename, "rb") as f:
        for line in f:
            json_line = json.loads(line)
            image_filenames.append(json_line["image_path"])
            annotations = json_line["classification_annotations"]
            labels = []
            for annotation in annotations:
                if annotation["annotation_label"] in all_labels:
                    labels = [annotation["annotation_label"]]
            image_labels.append(labels)
    return image_filenames, image_labels


def get_rounded_number(val: float) -> float:
    """Rounds a number and handles NaN/Inf values."""
    if np.isnan(val) or np.isinf(val):
        return -1
    else:
        return float(round(val, ROUNDING_DIGITS))


def save_validation_metrics(metrics: dict, output_dir: str) -> None:
    """Saves validation metrics to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, metrics_filename)
    with open(filename, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"Validation metrics saved to {filename}")


def save_confusion_matrix(y_true, y_pred, labels, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")


def preprocess_image(filename: str, image_size: ty.Tuple[int, int]) -> np.ndarray:
    """
    Reads, decodes, and preprocesses an image to the format expected by the model.
    This is based on the preprocessing for EfficientNetB0, which is commonly used.
    """

    """Crops an image to a centered square using TensorFlow."""
    # Read and decode the image
    image_bytes = tf.io.read_file(filename)
    image = tf.io.decode_image(image_bytes, channels=3)

    # Convert to float for calculations
    image_float = tf.cast(image, tf.float32)

    # Get the original image dimensions
    original_height = tf.shape(image_float)[0]
    original_width = tf.shape(image_float)[1]

    # Determine the size of the largest possible square crop
    crop_size = tf.minimum(original_height, original_width)

    # Calculate the starting coordinates for the centered crop
    offset_height = (original_height - crop_size) // 2
    offset_width = (original_width - crop_size) // 2

    # Perform the crop using explicit bounding box coordinates
    cropped_image = tf.image.crop_to_bounding_box(
        image_float,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=crop_size,
        target_width=crop_size,
    )

    # Resize the cropped square to the target image_size
    resized_image = tf.image.resize(cropped_image, image_size)

    # Save the preprocessed image to disk for inspection
    debug_dir = os.path.join(os.path.dirname(filename), "debug_preprocessed")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, os.path.basename(filename))
    tf.keras.utils.save_img(debug_path, resized_image)
    print(f"Preprocessed image saved to {debug_path}")
    resized_image = tf.cast(resized_image, tf.float32)
    return resized_image.numpy()


if __name__ == "__main__":
    DATA_JSON, MODEL_DIR, OUTPUT_DIR = parse_args(sys.argv[1:])

    # 1. Load labels and the TFLite model
    labels = load_labels(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, "model.tflite")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    IMG_SIZE = (input_shape[1], input_shape[2])
    print(f"image input size: {IMG_SIZE}")
    print(f"TFLite model loaded with input shape: {input_shape}")
    print(f"Labels loaded: {labels}")

    # 2. Load and prepare the test dataset
    image_filenames, image_labels = parse_filenames_and_labels_from_json(
        DATA_JSON, labels
    )

    # Filter out images with unknown or missing labels
    valid_data = [
        (filename, label[0])
        for filename, label in zip(image_filenames, image_labels)
        if label
    ]

    if not valid_data:
        raise ValueError("No valid images found in the dataset for evaluation.")

    filenames, true_labels = zip(*valid_data)
    print(f"Loaded {len(filenames)} images for validation.")

    # Create integer encoder for labels
    label_to_id = {label: i for i, label in enumerate(labels)}
    true_labels_encoded = [label_to_id[label] for label in true_labels]

    # 3. Perform inference and collect predictions
    predicted_labels = []
    predicted_probs = []

    for filename in filenames:
        try:
            image = preprocess_image(filename, IMG_SIZE)
            interpreter.set_tensor(
                input_details[0]["index"], np.expand_dims(image, axis=0)
            )
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])

            if len(labels) == 2:  # Binary classification
                prob = float(output_data[0][0])
                predicted_probs.append(prob)
                predicted_label_index = int(prob > 0.62)
            else:  # Multi-class classification
                predicted_label_index = int(np.argmax(output_data))

            predicted_labels.append(predicted_label_index)
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            predicted_labels.append(-1)
            if len(labels) == 2:
                predicted_probs.append(0.0)

    # Filter out failed predictions
    valid_indices = [i for i, pred in enumerate(predicted_labels) if pred != -1]
    filtered_true_labels = [true_labels_encoded[i] for i in valid_indices]
    filtered_predicted_labels = [predicted_labels[i] for i in valid_indices]

    if not filtered_true_labels:
        print("No successful predictions were made. Cannot calculate metrics.")
        sys.exit(1)

    # 4. Calculate and save metrics
    metrics = {}

    if len(labels) == 2:  # Binary Classification
        filtered_predicted_probs = [predicted_probs[i] for i in valid_indices]

        save_prediction_scores(
            [filenames[i] for i in valid_indices],
            filtered_true_labels,
            filtered_predicted_probs,
            OUTPUT_DIR,
        )

        metrics["accuracy"] = get_rounded_number(
            accuracy_score(filtered_true_labels, filtered_predicted_labels)
        )
        metrics["precision"] = get_rounded_number(
            precision_score(filtered_true_labels, filtered_predicted_labels)
        )
        metrics["recall"] = get_rounded_number(
            recall_score(filtered_true_labels, filtered_predicted_labels)
        )
        metrics["f1_score"] = get_rounded_number(
            f1_score(filtered_true_labels, filtered_predicted_labels)
        )

        plot_f1_vs_threshold(
            np.array(filtered_true_labels),
            np.array(filtered_predicted_probs),
            OUTPUT_DIR,
        )

    else:  # Multi-class Classification
        metrics["accuracy"] = get_rounded_number(
            accuracy_score(filtered_true_labels, filtered_predicted_labels)
        )
        metrics["precision"] = get_rounded_number(
            precision_score(
                filtered_true_labels,
                filtered_predicted_labels,
                average="weighted",
                zero_division=0,
            )
        )
        metrics["recall"] = get_rounded_number(
            recall_score(
                filtered_true_labels,
                filtered_predicted_labels,
                average="weighted",
                zero_division=0,
            )
        )
        metrics["f1_score"] = get_rounded_number(
            f1_score(
                filtered_true_labels,
                filtered_predicted_labels,
                average="weighted",
                zero_division=0,
            )
        )

    save_validation_metrics(metrics, OUTPUT_DIR)
    save_confusion_matrix(
        filtered_true_labels, filtered_predicted_labels, labels, OUTPUT_DIR
    )
