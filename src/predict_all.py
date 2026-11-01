from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import yaml
from src.data import load_metadata, make_dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
def main():
    with open(PROJECT_ROOT / "config.yaml", "r",encoding="utf-8") as file:
        config = yaml.safe_load(file)
    metadata = load_metadata(
        PROJECT_ROOT/config["data"]["metadata_path"]
    )
    all_dataset = make_dataset(
        metadata,
        config["data"]["image_size"],
        config["data"]["batch_size"],
        shuffle=False,
    )
    model = tf.keras.models.load_model(
        PROJECT_ROOT / config["model"]["path"]
    )
    probabilities = model.predict(all_dataset).ravel()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = results_dir / "selected_threshold.json"
    if threshold_path.exists():
        with open(threshold_path, "r",encoding="utf-8") as file:
            threshold = float(json.load(file)["threshold"])
    else:
        threshold = config["model"]["threshold"]
    results = metadata.copy()
    results["mass_nodule_probability"] = probabilities
    results["predicted_label"] = (
        probabilities >= threshold
    ).astype(int)
    results["screening_flag"] = np.where(
        results["predicted_label"] == 1,
        "Needs expert review",
        "Lower probability",
    )
    results.to_csv(
        results_dir / "all_screening_results.csv",
        index=False,
    )
    print(f"Screened {len(results):,} images")
    print(results["screening_flag"].value_counts())

if __name__ == "__main__":
    main()