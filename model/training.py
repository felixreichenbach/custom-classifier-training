import argparse
import json
import os
import typing as ty

import numpy as np
import tensorflow as tf
from keras import Model, callbacks
from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils


single_label = "MODEL_TYPE_SINGLE_LABEL_CLASSIFICATION"
multi_label = "MODEL_TYPE_MULTI_LABEL_CLASSIFICATION"
object_detection = "MODEL_TYPE_OBJECT_DETECTION"
tensorflow_framework = "MODEL_FRAMEWORK_TENSORFLOW"
tflite_framework = "MODEL_FRAMEWORK_TFLITE"
metrics_filename = "model_metrics.json"
labels_filename = "labels.txt"
default_prod_ml_training_bucket = "viam-ml-training"
default_stg_ml_training_bucket = "viam-staging-ml-training"

TFLITE_OPS = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]

TFLITE_OPTIMIZATIONS = [tf.lite.Optimize.DEFAULT]

ROUNDING_DIGITS = 5

# Normalization parameters are required when reprocessing the image.
_INPUT_NORM_MEAN = 127.5
_INPUT_NORM_STD = 127.5
_INPUT_MIN = 0
_INPUT_MAX = 255

detections_metrics_abbr = {
    "percent_boxes_matched_with_anchor": "pct_boxes_matched",
    "classification_loss": "class_loss",
}
early_stopping_key = "early_stopping"
reduce_lr_on_plateau_key = "reduce_lr_on_plateau"
early_stopping_monitor_val = {
    single_label: "categorical_accuracy",
    multi_label: "binary_accuracy",
    object_detection: "val_loss",
}


def get_neural_network_params(
    num_classes: int, model_type: str
) -> ty.Tuple[str, str, str, str]:
    """Function that returns units and activation used for the last layer
        and loss function for the model, based on number of classes and model type.
    Args:
        labels: list of labels corresponding to images
        model_type: string single-label or multi-label for desired output
    """
    # Single-label Classification
    if model_type == single_label:
        units = num_classes
        activation = "softmax"
        loss = tf.keras.losses.categorical_crossentropy
        metrics = (
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        )
    # Multi-label Classification
    elif model_type == multi_label:
        units = num_classes
        activation = "sigmoid"
        loss = tf.keras.losses.binary_crossentropy
        metrics = (
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        )
    return units, activation, loss, metrics


def get_callbacks(model_type: str):
    # Single-label Classification or Multi-label Classification
    if model_type == single_label or model_type == multi_label:
        callbackEarlyStopping = tf.keras.callbacks.EarlyStopping(
            # Stop training when `monitor` value is no longer improving
            monitor=early_stopping_monitor_val[model_type],
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
    # Object detection
    else:
        callbackEarlyStopping = tf.keras.callbacks.EarlyStopping(
            # Stop training when `monitor` value is no longer improving
            monitor=early_stopping_monitor_val[model_type],
            # "no longer improving" being defined as "no better than 'min_delta' less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 'patience' epochs"
            patience=3,
        )
        callbackReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
            # Reduce learning rate when `loss` is no longer improving
            monitor="loss",
            # "no longer improving" being defined as "no better than 'min_delta' less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 'patience' epochs"
            patience=5,
            # Default lower bound on learning rate
            min_lr=0,
        )
    return {
        early_stopping_key: callbackEarlyStopping,
        reduce_lr_on_plateau_key: callbackReduceLROnPlateau,
    }


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
                img_size[0], img_size[1], crop_to_aspect_ratio=False
            ),
        ]
    )
    return preprocessing


# Build the Keras model
def build_and_compile_classification(
    labels: ty.List[str], model_type: str, input_shape: ty.Tuple[int, int, int]
) -> Model:
    units, activation, loss_fnc, metrics = get_neural_network_params(
        len(labels), model_type
    )

    x = tf.keras.Input(input_shape, dtype=tf.uint8)
    # Data processing
    preprocessing = preprocessing_layers_classification(input_shape[:-1])
    data_augmentation = tf.keras.Sequential(
        [
            # tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    # Get the pre-trained model
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    # Add custom layers
    global_pooling = tf.keras.layers.GlobalAveragePooling2D()
    # Output layer
    classification = tf.keras.layers.Dense(units, activation=activation, name="output")

    y = tf.keras.Sequential(
        [
            preprocessing,
            data_augmentation,
            base_model,
            global_pooling,
            classification,
        ]
    )(x)

    model = tf.keras.Model(x, y)

    model.compile(
        loss=loss_fnc,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[metrics],
    )
    return model


def save_model_metrics_classification(
    loss_history: tf.keras.callbacks.History,
    monitored_val: ty.List[str],
    model_dir: str,
    model: Model,
    test_dataset: tf.data.Dataset,
) -> None:
    test_images = np.array([x for x, _ in test_dataset])
    test_labels = np.array([y for _, y in test_dataset])

    test_metrics = model.evaluate(test_images, test_labels)

    metrics = {}
    # Since there could be potentially many occurences of the maximum value being monitored,
    # we reverse the list storing the tracked values and take the last occurence.
    monitored_metric_max_idx = len(monitored_val) - np.argmax(monitored_val[::-1]) - 1
    for i, key in enumerate(model.metrics_names):
        metrics["train_" + key] = get_rounded_number(
            loss_history.history[key][monitored_metric_max_idx], ROUNDING_DIGITS
        )
        metrics["test_" + key] = get_rounded_number(test_metrics[i], ROUNDING_DIGITS)

    # Save the loss and test metrics as model metrics
    filename = os.path.join(model_dir, metrics_filename)
    with open(filename, "w") as f:
        json.dump(metrics, f, ensure_ascii=False)


def get_rounded_number(val: tf.Tensor, rounding_digits: int) -> tf.Tensor:
    if np.isnan(val) or np.isinf(val):
        return -1
    else:
        return float(round(val, rounding_digits))


def save_labels(labels: ty.List[str], model_dir: str) -> None:
    filename = os.path.join(model_dir, labels_filename)
    with open(filename, "w") as f:
        for label in labels[:-1]:
            f.write(label + "\n")
        f.write(labels[-1])


def save_classification_model(
    model: Model,
    model_dir: str,
    model_name: str,
    target_shape: ty.Tuple[int, int, int],
    is_tensorflow: bool,
) -> None:
    # Convert the model to tflite, with batch size 1 so the graph does not have dynamic-sized tensors.
    input = tf.keras.Input(target_shape, batch_size=1, dtype=tf.uint8)
    output = model(input, training=False)
    wrapped_model = tf.keras.Model(inputs=input, outputs=output)
    if is_tensorflow:
        # Save the model to GCS
        tf.saved_model.save(wrapped_model, model_dir)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(wrapped_model)
        converter.target_spec.supported_ops = TFLITE_OPS
        tflite_model = converter.convert()

        ImageClassifierWriter = image_classifier.MetadataWriter
        # Task Library expects label files that are in the same format as the one below.
        labels_file = os.path.join(model_dir, labels_filename)

        # Create the metadata writer.
        writer = ImageClassifierWriter.create_for_inference(
            tflite_model, [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [labels_file]
        )

        filename = os.path.join(model_dir, f"{model_name}.tflite")
        # Populate the metadata into the model.
        # Save the model to GCS
        writer_utils.save_file(writer.populate(), filename)


def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", dest="model_type", type=str)
    parser.add_argument("--labels", dest="labels", nargs="+", type=str)
    parser.add_argument("--model_dir", dest="model_dir", type=str)
    parser.add_argument("--data_json", dest="data_json", type=str)
    parser.add_argument("--model_name", dest="model_name", type=str)
    parser.add_argument("--model_framework", dest="model_framework", type=str)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int)
    args = parser.parse_args()
    return args


def parse_filenames_and_labels_from_json(
    filename: str,
    model_type: str,
) -> ty.Tuple[ty.List[str], ty.List[str]]:
    """Load and parse JSON file to return image filenames and corresponding labels.
    Args:
        filename: JSONLines file containing filenames and labels
        model_type: either 'single_label' or 'multi_label'
    """
    image_filenames = []
    image_labels = []

    with open(filename, "rb") as f:
        for line in f:
            json_line = json.loads(line)
            image_filenames.append(os.path.join("/gcs", json_line["image_gcs_uri"]))
            if model_type == single_label:
                labels = [json_line["classification_annotation"]["annotation_label"]]
            elif model_type == multi_label:
                annotations = json_line["classification_annotations"]
                labels = [annotation["annotation_label"] for annotation in annotations]
            image_labels.append(labels)
    return image_filenames, image_labels


def encoded_labels(
    image_labels: ty.List[str], all_labels: ty.List[str], model_type: str
) -> tf.Tensor:
    if model_type == single_label:
        encoder = tf.keras.layers.StringLookup(
            vocabulary=all_labels, num_oov_indices=0, output_mode="one_hot"
        )
    elif model_type == multi_label:
        encoder = tf.keras.layers.StringLookup(
            vocabulary=all_labels, num_oov_indices=0, output_mode="multi_hot"
        )
    return encoder(image_labels)


def create_dataset_classification(
    filenames: ty.List[str],
    labels: ty.List[str],
    all_labels: ty.List[str],
    model_type: str,
    img_size: ty.Tuple[int, int] = (256, 256),
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
    """
    # Create a first dataset of file paths and labels
    if model_type == single_label:
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (filenames, tf.ragged.constant(labels))
        )

    def mapping_fnc(x, y):
        return parse_image_and_encode_labels(x, y, all_labels, model_type, img_size)

    # Parse and preprocess observations in parallel
    dataset = dataset.map(mapping_fnc, num_parallel_calls=num_parallel_calls)

    # Shuffle the data for each buffer size
    # Disabling reshuffling ensures items from the training and test set will not get shuffled into each other
    dataset = dataset.shuffle(
        buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False
    )

    train_size = int(train_split * len(filenames))

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    # Batch the data for multiple steps
    # If the size of training data is smaller than the batch size,
    # batch the data to expand the dimensions by a length 1 axis.
    # This will ensure that the training data is valid model input
    train_batch_size = batch_size if batch_size < train_size else train_size
    if model_type == single_label:
        train_dataset = train_dataset.batch(train_batch_size)
    else:
        train_dataset = train_dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(train_batch_size)
        )

    # Fetch batches in the background while the model is training.
    train_dataset = train_dataset.prefetch(buffer_size=prefetch_buffer_size)

    return train_dataset, test_dataset


def decode_image(image):
    """Decodes the image as an uint8 dense vector
    Args:
        image: the image file contents as a tensor
    """
    return tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
        dtype=tf.dtypes.uint8,
    )


def check_type_and_decode_image(image_string_tensor):
    """Parse an image from gcs and decode it. Ungzip the image from gcs if zipped
    Args:
        image_string_tensor: the tensored form of an image gcs string
    """
    # Read an image from gcs
    image_string = tf.io.read_file(image_string_tensor)
    # Check file name if gzipped
    split_string = tf.strings.split(image_string_tensor, ".")

    if split_string[-1] == "gz":
        # Decode image from .GZ
        image_decompressed = tf.io.decode_compressed(
            image_string, compression_type="GZIP"
        )
        return decode_image(image_decompressed)

    return decode_image(image_string)


def parse_image_and_encode_labels(
    filename: str,
    labels: ty.List[str],
    all_labels: ty.List[str],
    model_type: str,
    img_size: ty.Tuple[int, int] = (256, 256),
) -> ty.Tuple[tf.Tensor, tf.Tensor]:
    """Returns a tuple of normalized image array and hot encoded labels array.
    Args:
        filename: string representing path to image
        labels: list of up to N_LABELS associated with image
        all_labels: list of all N_LABELS
        model_type: string single_label or multi_label
    """
    image_decoded = check_type_and_decode_image(filename)

    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [img_size[0], img_size[1]])
    # Convert string labels to encoded labels
    labels_encoded = encoded_labels(labels, all_labels, model_type)
    return image_resized, labels_encoded


if __name__ == "__main__":

    # Set up compute device strategy
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

    args = parse_args()
    MODEL_DIR = args.model_dir
    DATA_JSON = args.data_json
    MODEL_NAME = args.model_name
    EPOCHS = 50 if args.num_epochs is None or 0 else int(args.num_epochs)
    if EPOCHS < 0:
        raise ValueError("Invalid number of epochs, must be a positive nonzero number")

    # Parse arguments
    # args = parse_args()
    # MODEL_DIR = args.model_dir
    # DATA_JSON = args.data_json
    LABELS = ["OK", "NOK"]  # [label for label in args.labels[0].split(",")]
    MODEL_TYPE = single_label  # args.model_type -> fixed to single_label
    # MODEL_NAME = args.model_name
    path_to_model_dir = MODEL_DIR  # os.path.join(BUCKET_NAME, MODEL_DIR)
    MODEL_FRAMEWORK = (
        tflite_framework  # args.model_framework -> fixed to tflite_framework
    )

    if not (
        MODEL_FRAMEWORK == tensorflow_framework or MODEL_FRAMEWORK == tflite_framework
    ):
        exit("invalid model framework specified")

    is_tensorflow = MODEL_FRAMEWORK == tensorflow_framework

    # Dataset constants
    if MODEL_TYPE == single_label or MODEL_TYPE == multi_label:
        BATCH_SIZE = 16
        IMG_SIZE = (256, 256)
        SHUFFLE_BUFFER_SIZE = 32  # Shuffle the training data by a chunk of observations
    else:
        raise TypeError("invalid model type")
    AUTOTUNE = (
        tf.data.experimental.AUTOTUNE
    )  # Adapt preprocessing and prefetching dynamically

    # Model constants
    NUM_WORKERS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
    EPOCHS = 1000

    if MODEL_TYPE == single_label or MODEL_TYPE == multi_label:
        # Get filenames and labels of all images
        image_filenames, image_labels = parse_filenames_and_labels_from_json(
            filename=DATA_JSON,
            model_type=MODEL_TYPE,
        )

        # Generate 80/20 split for train and test data
        train_dataset, test_dataset = create_dataset_classification(
            filenames=image_filenames,
            labels=image_labels,
            all_labels=LABELS,
            model_type=MODEL_TYPE,
            img_size=IMG_SIZE,
            train_split=0.8,
            batch_size=GLOBAL_BATCH_SIZE,
            shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
            num_parallel_calls=AUTOTUNE,
            prefetch_buffer_size=AUTOTUNE,
        )

        # Build and compile model
        with strategy.scope():
            model = build_and_compile_classification(
                LABELS, MODEL_TYPE, IMG_SIZE + (3,)
            )
        # Get callbacks for training classification
        callbacks = get_callbacks(model_type=MODEL_TYPE)
        # Train model on data
        loss_history = model.fit(
            x=train_dataset, epochs=EPOCHS, callbacks=callbacks.values()
        )
        # Get the values of what is being monitored in the early stopping policy,
        # since this is what is used to restore best weights for the resulting model.
        monitored_val = callbacks[early_stopping_key].get_monitor_value(
            loss_history.history
        )
        # Save trained model metrics to JSON file
        save_model_metrics_classification(
            loss_history,
            monitored_val,
            path_to_model_dir,
            model,
            test_dataset,
        )
        # Save labels.txt file
        save_labels(LABELS, path_to_model_dir)
        # Save model and convert to TFLite if needed
        save_classification_model(
            model, path_to_model_dir, MODEL_NAME, IMG_SIZE + (3,), is_tensorflow
        )
    else:
        exit("invalid model type specified")
