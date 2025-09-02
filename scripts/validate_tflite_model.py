import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess_image(image_path, input_shape):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img, dtype=np.float32)
    # Normalize if needed (0-1)
    if np.max(img_array) > 1.0:
        img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_inference(interpreter, input_details, output_details, img_array):
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data.tolist()


def validate_model(model_path, images_dir, output_json):
    interpreter, input_details, output_details = load_tflite_model(model_path)
    input_shape = input_details[0]["shape"]
    results = {}
    for filename in os.listdir(images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            image_path = os.path.join(images_dir, filename)
            img_array = preprocess_image(image_path, input_shape)
            output = run_inference(
                interpreter, input_details, output_details, img_array
            )
            results[filename] = output
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate TFLite model on images.")
    parser.add_argument("--model", required=True, help="Path to model.tflite")
    parser.add_argument("--images", required=True, help="Directory of images")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    validate_model(args.model, args.images, args.output)
