from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
)
from src.data import load_metadata, make_dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn) if tp + fn else 0),
        "specificity": float(tn / (tn + fp) if tn + fp else 0),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
def main():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    metadata = load_metadata(
        PROJECT_ROOT / config["data"]["metadata_path"]
    )
    validation_df = metadata[
        metadata["split"] == "validation"
    ].reset_index(drop=True)
    validation_dataset = make_dataset(
        validation_df,
        config["data"]["image_size"],
        config["data"]["batch_size"],
        shuffle=False,
    )
    model = tf.keras.models.load_model(
        PROJECT_ROOT / config["model"]["path"]
    )
    probabilities = model.predict(validation_dataset).ravel()
    actual = validation_df["target"].to_numpy()
    rows = [
        calculate_metrics(actual, probabilities, threshold)
        for threshold in np.arange(0.05, 1.00, 0.05)
    ]
    threshold_frame = pd.DataFrame(rows)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    threshold_frame.to_csv(
        results_dir / "validation_thresholds.csv",
        index=False,
    )
    eligible = threshold_frame[
        threshold_frame["sensitivity"] >= 0.90
    ]
    if len(eligible) > 0:
        selected = eligible.sort_values(
            "specificity", ascending=False
        ).iloc[0]
    else:
        selected = threshold_frame.sort_values(
        ["sensitivity", "specificity"],
            ascending=False,
        ).iloc[0]
    selected_threshold = {
        "threshold": float(selected["threshold"]),
        "selection_rule": "Highest validation specificity with sensitivity >= 0.90 when possible",
        "validation_sensitivity": float(selected["sensitivity"]),
        "validation_specificity": float(selected["specificity"]),
    }
    with open(
        results_dir / "selected_threshold.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(selected_threshold, file, indent=2)
    print(selected_threshold)
if __name__ == "__main__":
    main()