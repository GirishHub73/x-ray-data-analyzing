from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from src.data import load_metadata, make_dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    )
    tn, fp, fn, tp = matrix.ravel()
    try:
        roc_auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        roc_auc = None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "sensitivity": float(tp / (tp + fn) if tp + fn else 0),
        "specificity": float(tn / (tn + fp) if tn + fp else 0),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
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
    test_df = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)
    test_dataset = make_dataset(
        test_df,
        config["data"]["image_size"],
        config["data"]["batch_size"],
        shuffle=False,
    )
    model = tf.keras.models.load_model(
        PROJECT_ROOT / config["model"]["path"]
    )
    probabilities = model.predict(test_dataset).ravel()
    actual = test_df["target"].to_numpy()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = results_dir / "selected_threshold.json"
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as file:
            threshold = float(json.load(file)["threshold"])
    else:
        threshold = config["model"]["threshold"]
    predictions = (probabilities >= threshold).astype(int)
    results = test_df.copy()
    results["mass_nodule_probability"] = probabilities
    results["predicted_label"] = predictions
    results["screening_flag"] = np.where(
        predictions == 1,
        "Needs expert review",
        "Lower probability",
    )
    results["correct"] = (
        results["target"] == results["predicted_label"]
    )
    results.to_csv(results_dir / "test_predictions.csv", index=False)
    results[
        (results["target"] == 1)
        & (results["predicted_label"] == 0)
    ].to_csv(results_dir / "false_negatives.csv", index=False)
    results[~results["correct"]].to_csv(
        results_dir / "incorrect_predictions.csv", index=False
    )
    metrics = calculate_metrics(actual, probabilities, threshold)
    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    report = classification_report(
        actual,
        predictions,
        target_names=["No Mass/Nodule", "Possible Mass/Nodule"],
        output_dict=True,
        zero_division=0,
    )
    with open(results_dir / "classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    matrix = confusion_matrix(actual, predictions, labels=[0, 1])
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No Finding", "Predicted Mass/Nodule"],
        yticklabels=["Actual No Finding", "Actual Mass/Nodule"],
    )
    plt.xlabel("Prediction")
    plt.ylabel("Actual label")
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=200)
    plt.close()
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, probabilities)
    plt.figure(figsize=(7, 6))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve.png", dpi=200)
    plt.close()
    comparison = []
    for candidate in np.arange(0.05, 1.00, 0.05):
        comparison.append(
            calculate_metrics(actual, probabilities, candidate)
        )
    pd.DataFrame(comparison).to_csv(
        results_dir / "test_threshold_comparison.csv",
        index=False,
    )
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__":
    main()