import argparse
import json
import os
import sys
import typing as ty
import tensorflow as tf
from keras import Model, callbacks
import numpy as np
import shutil
from keras.applications import EfficientNetB0
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
import keras

labels_filename = "labels.txt"
unknown_label = "UNKNOWN"
metrics_filename = "model_metrics.json"


TFLITE_OPS = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]

ROUNDING_DIGITS = 5


def parse_args(args):
    """Returns dataset file, model output directory, and num_epochs if present. These must be parsed as command line
    arguments and then used as the model input and output, respectively. The number of epochs can be used to optionally override the default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", dest="data_json", type=str)
    parser.add_argument("--model_output_directory", dest="model_dir", type=str)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int)
    parser.add_argument(
        "--labels",
        dest="labels",
        type=str,
        required=False,
        help="Space-separated list of labels, MUST be enclosed in single quotes",
        # ex: 'green_square blue_triangle'
    )
    parsed_args = parser.parse_args(args)
    return (
        parsed_args.data_json,
        parsed_args.model_dir,
        parsed_args.num_epochs,
        parsed_args.labels,
    )


def parse_filenames_and_labels_from_json(
    filename: str,
    all_labels: ty.List[str],
) -> ty.Tuple[ty.List[str], ty.List[str]]:
    """Load and parse JSON file to return image filenames and corresponding labels.
       The JSON file contains lines, where each line has the key "image_path" and "classification_annotations".
    Args:
        filename: JSONLines file containing filenames and labels
        all_labels: list of all N_LABELS
    """

    # TODO: Simplify for single label models
    image_filenames = []
    image_labels = []

    with open(filename, "rb") as f:
        for line in f:
            json_line = json.loads(line)
            image_filenames.append(json_line["image_path"])

            annotations = json_line["classification_annotations"]
            labels = [unknown_label]
            for annotation in annotations:
                # For single label model, we want at most one label.
                # If multiple valid labels are present, we arbitrarily select the last one.
                if annotation["annotation_label"] in all_labels:
                    labels = [annotation["annotation_label"]]
            image_labels.append(labels)
    return image_filenames, image_labels


def save_jsonl(model_dir: str, filename: str) -> None:
    """Copies the JSONLines file to the specified model directory."""
    dest = os.path.join(model_dir, "parsed_json_lines.jsonl")
    shutil.copyfile(filename, dest)


def preprocessing_layers_classification(
    img_size: ty.Tuple[int, int] = (256, 256)
) -> ty.Tuple[tf.Tensor, tf.Tensor]:
    """Preprocessing steps to apply to all images passed through the model.
    Args:
        img_size: optional 2D shape of image
    """
    preprocessing = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(
                img_size[0],
                img_size[1],
                crop_to_aspect_ratio=True,  # Changed to True to center crop before resizing
            ),
        ]
    )
    return preprocessing


def encoded_labels(
    image_labels: ty.List[str],
    all_labels: ty.List[str],
) -> tf.Tensor:
    """Returns a tuple of normalized image array and hot encoded labels array.
    Args:
        image_labels: labels present in image
        all_labels: list of all N_LABELS
        model_type: string single_label or multi_label
    """

    encoder = tf.keras.layers.StringLookup(
        vocabulary=all_labels, num_oov_indices=0, output_mode="one_hot"
    )
    return encoder(image_labels)


def parse_image_and_encode_labels(
    filename: str,
    labels: ty.List[str],
    all_labels: ty.List[str],
) -> ty.Tuple[tf.Tensor, tf.Tensor]:
    """Returns a tuple of normalized image array and hot encoded labels array.
    Args:
        filename: string representing path to image
        labels: list of up to N_LABELS associated with image
        all_labels: list of all N_LABELS
        model_type: string single_label or multi_label
        img_size: optional 2D shape of image
    """
    image_bytes = tf.io.read_file(filename)
    image_decoded = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
        dtype=tf.dtypes.uint8,
    )

    # Convert string labels to encoded labels
    labels_encoded = encoded_labels(labels, all_labels)
    return image_decoded, labels_encoded


def create_dataset(
    filenames,
    labels,
    all_labels,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
):
    # Ensure that there is at least one image in the dataset
    if len(filenames) == 0:
        raise ValueError("No images found in the dataset.")
    if len(filenames) != len(labels):
        raise ValueError(
            "Filenames and labels must have the same length. Found "
            f"{len(filenames)} filenames and {len(labels)} labels."
        )
    # Create a first dataset of file paths and labels

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # Apply a map to the dataset that converts filenames and text labels
    # to normalized images and encoded labels, respectively.
    def mapping_fnc(x, y):
        return parse_image_and_encode_labels(x, y, all_labels)

    # Parse and preprocess observations in parallel
    dataset = dataset.map(mapping_fnc, num_parallel_calls=num_parallel_calls)
    return dataset


def create_dataset_classification(
    filenames: ty.List[str],
    labels: ty.List[str],
    all_labels: ty.List[str],
    train_split: float = 0.8,
    batch_size: int = 64,
    shuffle_buffer_size: int = 1024,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    prefetch_buffer_size: int = tf.data.experimental.AUTOTUNE,
) -> ty.Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load and parse dataset from Tensorflow datasets.
    Args:
        filenames: string list of image paths
        labels: list of string lists, where each string list contains up to N_LABEL labels associated with an image
        all_labels: string list of all N_LABELS
        model_type: string single_label or multi_label
        img_size: optional 2D shape of image
        train_split: optional float between 0.0 and 1.0 to specify proportion of images that will be used for training
        batch_size: optional size for number of samples for each training iteration
        shuffle_buffer_size: optional size for buffer that will be filled and randomly sampled from, with replacement
        num_parallel_calls: optional integer representing the number of batches to compute asynchronously in parallel
        prefetch_buffer_size: optional integer representing the number of batches that will be buffered when prefetching
    """

    print("*********** Creating balanced datasets ***********")
    image_names_ok = []
    image_labels_ok = []
    image_names_nok = []
    image_labels_nok = []
    for i, filename in enumerate(filenames):
        # print(labels[i][0])
        match labels[i][0]:
            case "NOK":
                image_names_nok.append(filename)
                image_labels_nok.append(labels[i])
            case "OK":
                image_names_ok.append(filename)
                image_labels_ok.append(labels[i])
            case _:
                print(
                    f"Image label did not match: {filename}, removing from dataset or Label is not OK"
                )
    print(f"Number of images with label NOK: {len(image_names_nok)}")
    print(f"Number of labels with NOK: {len(image_labels_nok)}")
    print(f"Number of images with label OK: {len(image_names_ok)}")
    print(f"Number of labels with OK: {len(image_labels_ok)}")

    # Create datasets for each class as otherwise the split may introduce bias towards one class
    dataset_ok = create_dataset(
        image_names_ok,
        image_labels_ok,
        all_labels,
        num_parallel_calls,
    )
    dataset_nok = create_dataset(
        image_names_nok,
        image_labels_nok,
        all_labels,
        num_parallel_calls,
    )

    print(f"Length of OK dataset: {len(list(dataset_ok))}")
    print(f"Length of NOK dataset: {len(list(dataset_nok))}")

    train_size_ok = int(train_split * len(image_names_ok))
    train_size_nok = int(train_split * len(image_names_nok))
    train_size = train_size_ok + train_size_nok
    print(f"Training dataset size: {train_size} images")

    train_dataset = dataset_ok.take(train_size_ok)
    train_dataset = train_dataset.concatenate(dataset_nok.take(train_size_nok))
    test_dataset = dataset_ok.skip(train_size_ok)
    test_dataset = test_dataset.concatenate(dataset_nok.skip(train_size_nok))

    # Shuffle the data for each buffer size
    # Disabling reshuffling ensures items from the training and test set will not get shuffled into each other
    train_dataset = train_dataset.shuffle(
        buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False
    )

    test_batch_size = (
        batch_size
        if batch_size < (len(filenames) - train_size)
        else (len(filenames) - train_size)
    )

    test_dataset = test_dataset.batch(test_batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=prefetch_buffer_size)

    # Batch the data for multiple steps
    # If the size of training data is smaller than the batch size,
    # batch the data to expand the dimensions by a length 1 axis.
    # This will ensure that the training data is valid model input
    train_batch_size = batch_size if batch_size < train_size else train_size
    train_dataset = train_dataset.batch(train_batch_size)

    # Fetch batches in the background while the model is training.
    train_dataset = train_dataset.prefetch(buffer_size=prefetch_buffer_size)

    return train_dataset, test_dataset


# Build the Keras model
def build_classification_model(
    input_shape: ty.Tuple[int, int, int],
    num_classes: int,
    activation: str,
    dropout_rate: float = 0.2,
) -> Model:
    """
    Builds and compiles a classification model for fine-tuning using EfficientNetB0.

    This function defines the core model architecture. It's designed to work with a
    separate data pipeline that handles preprocessing and augmentation.

    Args:
        input_shape: A tuple representing the shape of the input images, e.g., (224, 224, 3).
        num_classes: The number of classes for the classification task. This determines the
                     number of units in the final output layer.
        activation: The activation function for the final output layer. For multi-class
                    classification, 'softmax' is common. For multi-label, 'sigmoid' is used.
        dropout_rate: The dropout rate for the regularization layer. A value between 0 and 1.

    Returns:
        A compiled Keras Model ready for training.
    """

    # Define the input layer with a float32 data type, which is a best practice.
    inputs = Input(shape=input_shape)

    # Load the pre-trained EfficientNetB0 model without its top layers,
    # and specify the input tensor.
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs
    )

    # Freeze the weights of the base model to prevent them from being updated
    # during training. This is a crucial step for transfer learning.
    base_model.trainable = False

    # Build the custom classification head using the Keras Functional API.
    x = base_model.output

    # Add custom layers on top of the base model
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation=activation, name="output")(x)

    # Create the complete model by defining the inputs and outputs.
    model = Model(inputs=base_model.input, outputs=outputs)

    return base_model, model


def create_data_pipeline(
    dataset: tf.data.Dataset,
    image_size: ty.Tuple[int, int],
    batch_size: int,
    is_training: bool = False,
) -> tf.data.Dataset:
    """
    Creates a data pipeline for preprocessing and optionally augmenting images.

    This function handles decoding, resizing, and normalization of images,
    along with optional data augmentation for the training set.

    Args:
        dataset: The raw tf.data.Dataset of images.
        image_size: A tuple representing the target size for the images (height, width).
        batch_size: The number of elements to combine in each batch.
        is_training: A boolean flag to determine whether to apply data augmentation.

    Returns:
        A preprocessed and batched tf.data.Dataset.
    """

    # Preprocessing and normalization function
    def preprocess_image(image, label):
        image = tf.image.resize(image, image_size)
        return image, label

    # Augmentation function for the training set
    def augment_image(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
        image = tf.image.random_brightness(image, max_delta=0.1)
        # Note: You can add more augmentation layers here as needed.
        return image, label

    preprocessing_pipeline = keras.Sequential(
        [
            keras.layers.Resizing(
                image_size[0], image_size[1], crop_to_aspect_ratio=True
            ),
            # keras.layers.Rescaling(1.0 / 255), -> Handled in TFLite model afaik
        ]
    )

    augmentation_pipeline = keras.Sequential(
        [
            keras.layers.RandomRotation(0.1),
            # keras.layers.RandomFlip("horizontal_and_vertical"),
            # keras.layers.RandomContrast(0.1),
            # keras.layers.RandomBrightness(0.1),
        ]
    )

    # Create the pipeline
    if is_training:
        dataset = dataset.map(
            lambda x, y: (preprocessing_pipeline(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(
            lambda x, y: (augmentation_pipeline(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda x, y: (preprocessing_pipeline(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def fine_tune_model(my_model: Model) -> Model:
    """
    Prepares a pre-trained model for fine-tuning by unfreezing layers.

    Args:
        model: The Keras Model to fine-tune. This model should have a frozen
               base model as a layer.
        fine_tune_from_layer: The index of the layer in the base model from
                              which to begin unfreezing. All layers with an
                              index greater than or equal to this will be
                              unfrozen.

    Returns:
        The recompiled Keras Model ready for the fine-tuning stage.
    """

    my_model.trainable = True
    for layer in my_model.layers[
        :-3
    ]:  # Unfreeze last 3 layers TODO: AI recommendation is to use -30
        layer.trainable = False
    # Recompile the model with a very low learning rate.
    # It's crucial to recompile the model for the changes to the trainable
    # state to take effect. Using a low learning rate prevents catastrophic
    # forgetting of the pre-trained weights.
    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=(
            tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ),
    )
    return my_model


def save_labels(labels: ty.List[str], model_dir: str) -> None:
    """Saves a label.txt of output labels to the specified model directory.
    Args:
        labels: list of string lists, where each string list contains up to N_LABEL labels associated with an image
        model_dir: output directory for model artifacts
    """
    filename = os.path.join(model_dir, labels_filename)
    with open(filename, "w") as f:
        for label in labels[:-1]:
            f.write(label + "\n")
        f.write(labels[-1])


def save_tflite_classification(
    model: Model,
    model_dir: str,
    model_name: str,
    target_shape: ty.Tuple[int, int, int],
) -> None:
    """Save model as a TFLite model.
    Args:
        model: trained model
        model_dir: output directory for model artifacts
        model_name: name of saved model
        target_shape: desired output shape of predictions from model
    """
    # Convert the model to tflite, with batch size 1 so the graph does not have dynamic-sized tensors.
    input = tf.keras.Input(target_shape, batch_size=1, dtype=tf.uint8)
    output = model(input, training=False)
    wrapped_model = tf.keras.Model(inputs=input, outputs=output)

    ## Working without TFLite-Support
    converter = tf.lite.TFLiteConverter.from_keras_model(wrapped_model)
    converter.target_spec.supported_ops = TFLITE_OPS
    tflite_model = converter.convert()
    filename = os.path.join(model_dir, f"{model_name}.tflite")
    # Writing the model buffer into a file.
    with open(filename, "wb") as f:
        f.write(tflite_model)


def get_rounded_number(val: tf.Tensor, rounding_digits: int) -> tf.Tensor:
    if np.isnan(val) or np.isinf(val):
        return -1
    else:
        return float(round(val, rounding_digits))


def save_model_metrics_classification(
    combined_history: dict,
    model_dir: str,
    model: Model,
    test_dataset: tf.data.Dataset,
) -> None:

    monitored_metric_key = "categorical_accuracy"

    monitored_val = combined_history[monitored_metric_key]

    # Find the index of the best value
    monitored_metric_max_idx = len(monitored_val) - np.argmax(monitored_val[::-1]) - 1

    test_metrics = model.evaluate(test_dataset)

    metrics = {}
    for i, key in enumerate(model.metrics_names):
        # Access metrics directly from the dictionary
        metrics["train_" + key] = get_rounded_number(
            combined_history[key][monitored_metric_max_idx],
            ROUNDING_DIGITS,
        )
        metrics["test_" + key] = get_rounded_number(test_metrics[i], ROUNDING_DIGITS)

    # Save the loss and test metrics as model metrics
    filename = os.path.join(model_dir, metrics_filename)
    with open(filename, "w") as f:
        json.dump(metrics, f, ensure_ascii=False)


def get_callbacks():
    """Returns callbacks for training classification model."""
    early_stopping_key = "early_stopping"
    reduce_lr_on_plateau_key = "reduce_lr_on_plateau"

    callbackEarlyStopping = tf.keras.callbacks.EarlyStopping(
        # Stop training when `monitor` value is no longer improving
        monitor="categorical_accuracy",
        # "no longer improving" being defined as "no better than 'min_delta' less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 'patience' epochs"
        patience=5,
        # Restore weights from the best performing model, requires keeping track of model weights and performance.
        restore_best_weights=True,
    )
    callbackReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        # Reduce learning rate when `loss` is no longer improving
        monitor="loss",
        # "no longer improving" being defined as "no better than 'min_delta' less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 'patience' epochs"
        patience=5,
        # Default lower bound on learning rate
        min_lr=0,
    )

    return {
        early_stopping_key: callbackEarlyStopping,
        reduce_lr_on_plateau_key: callbackReduceLROnPlateau,
    }


if __name__ == "__main__":
    # Set up compute device strategy. If GPUs are available, they will be used
    if len(tf.config.list_physical_devices("GPU")) > 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    IMG_SIZE = (256, 256)
    # Batch size, buffer size, epochs can be adjusted according to the training job.
    BATCH_SIZE = 16
    SHUFFLE_BUFFER_SIZE = 32
    AUTOTUNE = (
        tf.data.experimental.AUTOTUNE
    )  # Adapt preprocessing and prefetching dynamically

    # Model constants
    NUM_WORKERS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS

    DATA_JSON, MODEL_DIR, num_epochs, labels = parse_args(sys.argv[1:])
    EPOCHS = 200 if num_epochs is None or 0 else int(num_epochs)
    if EPOCHS < 0:
        raise ValueError("Invalid number of epochs, must be a positive nonzero number")

    # Read dataset file, labels should be changed according to the desired model output.
    LABELS = (
        ["orange_triangle", "blue_star"]
        if labels is None
        else [label for label in labels.strip("'").split()]
    )

    image_filenames, image_labels = parse_filenames_and_labels_from_json(
        DATA_JSON,
        LABELS,
    )
    # Generate 80/20 split for training and validation data
    train_dataset, val_dataset = create_dataset_classification(
        filenames=image_filenames,
        labels=image_labels,
        all_labels=LABELS + [unknown_label],
        train_split=0.8,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        num_parallel_calls=AUTOTUNE,
        prefetch_buffer_size=AUTOTUNE,
    )

    # Build and compile model
    with strategy.scope():

        num_classes = len(LABELS) + 1
        activation = "softmax"
        loss = tf.keras.losses.categorical_crossentropy
        metrics_names = (
            tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        )

        # Build model
        base_model, model = build_classification_model(
            (*IMG_SIZE, 3), num_classes, activation="softmax"
        )

        # Compile model
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=metrics_names,
        )

        print(model.summary())

        # Create the data pipelines
        train_data_pipeline = create_data_pipeline(
            train_dataset, IMG_SIZE, BATCH_SIZE, is_training=True
        )
        val_data_pipeline = create_data_pipeline(val_dataset, IMG_SIZE, BATCH_SIZE)

        # Get callbacks for training classification
        callbacks = get_callbacks()

        # Train the model
        loss_history = model.fit(
            train_data_pipeline,
            validation_data=val_data_pipeline,
            epochs=EPOCHS,
            callbacks=callbacks.values(),
        )

        # Fine-tuning the model
        # Prepare the model for fine-tuning by unfreezing layers.
        # The base model has 236 layers. Unfreezing from layer 100 is a good starting point.
        my_model = fine_tune_model(model)  # , fine_tune_from_layer=100)

        # Print the updated summary to see which layers are now trainable.
        my_model.summary()

        fine_tune_loss_history = my_model.fit(
            train_data_pipeline,
            validation_data=val_data_pipeline,
            epochs=EPOCHS + 5,
            initial_epoch=len(loss_history.epoch),
            callbacks=callbacks.values(),
        )

    # Create an empty dictionary to store the combined history
    combined_history = {}
    print("loss_history keys:", loss_history.history.keys())
    print("fine_tune_loss_history keys:", fine_tune_loss_history.history.keys())
    # Iterate over the keys (metrics) in the first history
    for key in loss_history.history.keys():
        # Concatenate the lists from both histories
        combined_history[key] = (
            loss_history.history[key] + fine_tune_loss_history.history[key]
        )

    # Save trained model metrics to JSON file
    save_model_metrics_classification(
        combined_history,
        MODEL_DIR,
        my_model,
        val_dataset,
    )

    # Save labels.txt file
    save_labels(LABELS + [unknown_label], MODEL_DIR)
    # Convert the model to tflite
    save_tflite_classification(model, MODEL_DIR, "model", (*IMG_SIZE, 3))

    save_jsonl(MODEL_DIR, DATA_JSON)
