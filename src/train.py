from pathlib import Path
import argparse
import json
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
from src.data import load_metadata, make_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="mass_nodule_probability",
    )(x)
    return models.Model(inputs, outputs), base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--fine-tune", action="store_true")
 args = parser.parse_args()
 with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
 tf.keras.utils.set_random_seed(config["seed"])
 metadata = load_metadata(
 PROJECT_ROOT / config["data"]["metadata_path"]
 )
 train_df = metadata[metadata["split"] == "train"].reset_index(drop=True)
 validation_df = metadata[
 metadata["split"] == "validation"
 ].reset_index(drop=True)
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
 results_dir = PROJECT_ROOT / config["results"]["directory"]
 models_dir = PROJECT_ROOT / "models"
 results_dir.mkdir(parents=True, exist_ok=True)
 models_dir.mkdir(parents=True, exist_ok=True)
 model_path = PROJECT_ROOT / config["model"]["path"]
 counts = train_df["target"].value_counts().to_dict()
 negative_count = counts.get(0, 1)
 positive_count = counts.get(1, 1)
 total = negative_count + positive_count
 class_weights = {
 0: total / (2.0 * negative_count),
 1: total / (2.0 * positive_count),
 }
 with open(results_dir / "class_weights.json", "w", encoding="utf-8") as file:
    json.dump(class_weights, file, indent=2)
 model, base_model = build_model(config["data"]["image_size"])
 compile_model(model, config["training"]["learning_rate"])
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
 CSVLogger(results_dir / "training_history.csv"),
 ]
 history = model.fit(
 train_dataset,
 validation_data=validation_dataset,
 epochs=config["training"]["epochs"],
 class_weight=class_weights,
 callbacks=callbacks,
 )
 pd.DataFrame(history.history).to_csv(
     results_dir / "training_history.csv",
     index=False,
 )
 if args.fine_tune:
     base_model.trainable = True
 for layer in base_model.layers[:-30]:
     layer.trainable = False
 for layer in base_model.layers:
     if isinstance(layer, layers.BatchNormalization):
         layer.trainable = False
 compile_model(
     model,
     config["training"]["fine_tune_learning_rate"],
 )
 model.fit(
     train_dataset,
     validation_data=validation_dataset,
     epochs=config["training"]["fine_tune_epochs"],
     class_weight=class_weights,
     callbacks=[
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
     ],
 )
 model.save(model_path)
 print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
