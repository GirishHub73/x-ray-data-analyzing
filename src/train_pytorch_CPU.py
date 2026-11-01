from pathlib import Path
import argparse
import json

import pandas as pd
import tensorflow as tf
import yaml

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from src.data import load_metadata, make_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_tensorflow_device():
    """Configure TensorFlow to use the GPU when available."""

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("TensorFlow GPU not detected.")
        print("Using CPU instead.")
        return "/CPU:0"

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        print(f"Could not configure GPU memory growth: {error}")

    print(f"TensorFlow detected {len(gpus)} GPU(s):")

    for gpu in gpus:
        print(f"  {gpu}")

    print("Using device: /GPU:0")

    return "/GPU:0"


def limit_training_images(dataframe, maximum_images, seed):
    """
    Limit the training data while preserving class balance.
    The validation dataset is not limited.
    """

    if maximum_images is None:
        return dataframe.reset_index(drop=True)

    if maximum_images <= 0:
        return dataframe.reset_index(drop=True)

    if len(dataframe) <= maximum_images:
        return dataframe.sample(
            frac=1,
            random_state=seed
        ).reset_index(drop=True)

    class_groups = list(dataframe.groupby("target"))

    if maximum_images < len(class_groups):
        raise ValueError(
            "maximum_images must be at least equal to the number of classes."
        )

    base_images_per_class = maximum_images // len(class_groups)
    remaining_images = maximum_images % len(class_groups)

    sampled_groups = []

    for index, (_, class_group) in enumerate(class_groups):
        number_to_sample = base_images_per_class

        if index < remaining_images:
            number_to_sample += 1

        number_to_sample = min(
            number_to_sample,
            len(class_group)
        )

        sampled_group = class_group.sample(
            n=number_to_sample,
            random_state=seed + index
        )

        sampled_groups.append(sampled_group)

    limited_dataframe = pd.concat(sampled_groups)

    # Fill any remaining slots if one class had fewer images than its quota.
    if len(limited_dataframe) < maximum_images:
        remaining_dataframe = dataframe.drop(
            limited_dataframe.index
        )

        additional_count = maximum_images - len(limited_dataframe)

        additional_samples = remaining_dataframe.sample(
            n=additional_count,
            random_state=seed
        )

        limited_dataframe = pd.concat(
            [limited_dataframe, additional_samples]
        )

    return limited_dataframe.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)


def build_model(image_size):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(0.02),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        name="efficientnetb0",
    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(image_size, image_size, 3)
    )

    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="mass_nodule_probability",
    )(x)

    model = models.Model(inputs, outputs)

    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(
                name="roc_auc"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune the final EfficientNet layers.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum number of training images. Default: 100.",
    )

    args = parser.parse_args()

    device_name = configure_tensorflow_device()

    with open(
        PROJECT_ROOT / "config.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    tf.keras.utils.set_random_seed(
        config["seed"]
    )

    metadata = load_metadata(
        PROJECT_ROOT
        / config["data"]["metadata_path"]
    )

    train_df = metadata[
        metadata["split"] == "train"
    ].reset_index(drop=True)

    validation_df = metadata[
        metadata["split"] == "validation"
    ].reset_index(drop=True)

    # Ensure target values are numeric.
    train_df["target"] = train_df["target"].astype(int)
    validation_df["target"] = validation_df["target"].astype(int)

    original_training_count = len(train_df)

    # Limit training data to 100 images while preserving class balance.
    train_df = limit_training_images(
        dataframe=train_df,
        maximum_images=args.max_images,
        seed=config["seed"],
    )

    print(
        f"Using {len(train_df)} of "
        f"{original_training_count} training images."
    )

    print(
        "Training class distribution:"
    )
    print(train_df["target"].value_counts().to_dict())

    print(
        f"Using {len(validation_df)} validation images."
    )

    train_dataset = make_dataset(
        train_df,
        config["data"]["image_size"],
        config["data"]["batch_size"],
        shuffle=True,
    )

    validation_dataset = make_dataset(
        validation_df,
        config["data"]["image_size"],
        config["data"]["batch_size"],
        shuffle=False,
    )

    results_dir = (
        PROJECT_ROOT
        / config["results"]["directory"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        PROJECT_ROOT
        / config["model"]["path"]
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = train_df["target"].value_counts().to_dict()

    negative_count = counts.get(0, 1)
    positive_count = counts.get(1, 1)

    total = negative_count + positive_count

    class_weights = {
        0: total / (2.0 * negative_count),
        1: total / (2.0 * positive_count),
    }

    with open(
        results_dir / "class_weights.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            class_weights,
            file,
            indent=2,
        )

    # Build and compile the model on the selected device.
    with tf.device(device_name):
        model, base_model = build_model(
            config["data"]["image_size"]
        )

        compile_model(
            model,
            config["training"]["learning_rate"],
        )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_roc_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_roc_auc",
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_roc_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    # Initial training.
    with tf.device(device_name):
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=config["training"]["epochs"],
            class_weight=class_weights,
            callbacks=callbacks,
        )

    pd.DataFrame(
        history.history
    ).to_csv(
        results_dir / "training_history.csv",
        index=False,
    )

    # Optional fine-tuning.
    if args.fine_tune:
        print("Starting fine-tuning...")

        base_model.trainable = True

        # Freeze all but the final 30 EfficientNet layers.
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # Keep batch normalization layers frozen.
        for layer in base_model.layers:
            if isinstance(
                layer,
                layers.BatchNormalization,
            ):
                layer.trainable = False

        with tf.device(device_name):
            compile_model(
                model,
                config["training"][
                    "fine_tune_learning_rate"
                ],
            )

        fine_tune_callbacks = [
            EarlyStopping(
                monitor="val_roc_auc",
                mode="max",
                patience=2,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_roc_auc",
                mode="max",
                save_best_only=True,
            ),
        ]

        with tf.device(device_name):
            fine_tune_history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=config["training"][
                    "fine_tune_epochs"
                ],
                class_weight=class_weights,
                callbacks=fine_tune_callbacks,
            )

        fine_tune_history_df = pd.DataFrame(
            fine_tune_history.history
        )

        fine_tune_history_df.to_csv(
            results_dir / "fine_tune_history.csv",
            index=False,
        )

    model.save(model_path)

    print(f"Saved model to: {model_path}")
    print(f"Training device: {device_name}")


if __name__ == "__main__":
    main()