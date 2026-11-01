from pathlib import Path
from contextlib import nullcontext
import argparse
import copy
import json
import random

import numpy as np
import pandas as pd
import yaml
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata_file(metadata_path):
    """Load metadata without importing TensorFlow."""

    metadata_path = Path(metadata_path)

    if metadata_path.suffix.lower() == ".csv":
        return pd.read_csv(metadata_path)

    if metadata_path.suffix.lower() == ".parquet":
        return pd.read_parquet(metadata_path)

    if metadata_path.suffix.lower() == ".json":
        try:
            return pd.read_json(metadata_path)
        except ValueError:
            return pd.read_json(metadata_path, lines=True)

    raise ValueError(
        f"Unsupported metadata format: {metadata_path.suffix}"
    )


def find_image_column(dataframe):
    possible_columns = [
        "image_path",
        "image_filepath",
        "image_file_path",
        "filepath",
        "file_path",
        "path",
        "filename",
        "file_name",
        "image",
    ]

    for column in possible_columns:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "Could not find an image path column. "
        f"Available columns: {list(dataframe.columns)}"
    )


def resolve_image_path(
    image_value,
    project_root,
    metadata_path,
    data_config,
):
    image_value = Path(str(image_value))

    configured_roots = []

    for key in [
        "image_root",
        "images_root",
        "image_directory",
        "images_directory",
    ]:
        configured_value = data_config.get(key)

        if configured_value:
            configured_root = Path(configured_value)

            if not configured_root.is_absolute():
                configured_root = project_root / configured_root

            configured_roots.append(configured_root)

    candidates = []

    if image_value.is_absolute():
        candidates.append(image_value)
    else:
        candidates.extend(
            [
                project_root / image_value,
                metadata_path.parent / image_value,
            ]
        )

        for root in configured_roots:
            candidates.append(root / image_value)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Return the first likely path so the error shows a useful location.
    return candidates[0] if candidates else image_value


def add_resolved_paths(
    dataframe,
    image_column,
    project_root,
    metadata_path,
    data_config,
):
    dataframe = dataframe.copy()

    dataframe["resolved_image_path"] = dataframe[
        image_column
    ].apply(
        lambda value: resolve_image_path(
            value,
            project_root,
            metadata_path,
            data_config,
        )
    )

    missing_paths = dataframe[
        ~dataframe["resolved_image_path"].apply(Path.exists)
    ]

    if not missing_paths.empty:
        examples = missing_paths[
            "resolved_image_path"
        ].head(5).tolist()

        raise FileNotFoundError(
            "Could not find one or more images. "
            f"Examples: {examples}"
        )

    return dataframe


def limit_training_images(dataframe, maximum_images, seed):
    """
    Select a balanced subset of the training data.

    For 100 images and two classes, this selects 50 images
    from each class when possible.
    """

    if maximum_images is None or maximum_images <= 0:
        return dataframe.reset_index(drop=True)

    if len(dataframe) <= maximum_images:
        return dataframe.sample(
            frac=1,
            random_state=seed,
        ).reset_index(drop=True)

    class_groups = list(dataframe.groupby("target"))

    if maximum_images < len(class_groups):
        raise ValueError(
            "max-images must be at least the number of classes."
        )

    images_per_class = maximum_images // len(class_groups)
    remainder = maximum_images % len(class_groups)

    selected_groups = []

    for index, (_, class_group) in enumerate(class_groups):
        number_to_select = images_per_class

        if index < remainder:
            number_to_select += 1

        number_to_select = min(
            number_to_select,
            len(class_group),
        )

        selected_groups.append(
            class_group.sample(
                n=number_to_select,
                random_state=seed + index,
            )
        )

    selected = pd.concat(selected_groups)

    # Fill remaining slots if a class did not have enough images.
    if len(selected) < maximum_images:
        remaining = dataframe.drop(selected.index)

        additional_count = maximum_images - len(selected)

        additional = remaining.sample(
            n=additional_count,
            random_state=seed,
        )

        selected = pd.concat([selected, additional])

    return selected.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class XRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = row["resolved_image_path"]
        label = float(row["target"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            label,
            dtype=torch.float32,
        )

        return image, label


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_model():
    weights = EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(
        weights=weights
    )

    # Freeze EfficientNet feature extractor initially.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        1,
    )

    return model


def prepare_for_fine_tuning(model):
    """
    Unfreeze the final EfficientNet feature stages while
    keeping earlier layers and batch normalization frozen.
    """

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Unfreeze the final two feature stages.
    final_feature_stages = list(model.features.children())[-2:]

    for stage in final_feature_stages:
        for parameter in stage.parameters():
            parameter.requires_grad = True

    # Keep batch-normalization layers frozen.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

            for parameter in module.parameters():
                parameter.requires_grad = False


def keep_batch_norm_frozen(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


# ---------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------

def calculate_metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(
            labels,
            probabilities,
        )
    else:
        roc_auc = 0.5

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }


def run_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
    scaler,
):
    is_training = optimizer is not None

    if is_training:
        model.train()
        keep_batch_norm_frozen(model)
    else:
        model.eval()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    use_amp = device.type == "cuda"

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=use_amp,
        )

        labels = labels.to(
            device,
            non_blocking=use_amp,
        )

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        autocast_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
            if use_amp
            else nullcontext()
        )

        with torch.set_grad_enabled(is_training):
            with autocast_context:
                logits = model(images).squeeze(1)
                loss = loss_function(logits, labels)

            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        probabilities = torch.sigmoid(logits)

        total_loss += loss.item() * images.size(0)

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )

    average_loss = total_loss / len(data_loader.dataset)

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_probabilities),
    )

    metrics["loss"] = average_loss

    return metrics


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    model_path,
    image_size,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "image_size": image_size,
    }

    torch.save(
        checkpoint,
        model_path,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    global outputs
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Unfreeze the final EfficientNet layers.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum training images. Use 0 for all images.",
    )

    args = parser.parse_args()

    config_path = PROJECT_ROOT / "config.yaml"

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        print(
            f"CUDA build: {torch.version.cuda}"
        )

        torch.backends.cudnn.benchmark = True

    data_config = config["data"]

    metadata_path = (
        PROJECT_ROOT
        / data_config["metadata_path"]
    )

    metadata = load_metadata_file(
        metadata_path
    )

    image_column = find_image_column(metadata)

    if "target" not in metadata.columns:
        raise ValueError(
            "Metadata must contain a 'target' column."
        )

    if "split" not in metadata.columns:
        raise ValueError(
            "Metadata must contain a 'split' column."
        )

    train_df = metadata[
        metadata["split"] == "train"
    ].copy()

    validation_df = metadata[
        metadata["split"] == "validation"
    ].copy()

    if train_df.empty:
        raise ValueError(
            "No training records were found."
        )

    if validation_df.empty:
        raise ValueError(
            "No validation records were found."
        )

    # Ensure binary numeric labels.
    train_df["target"] = train_df[
        "target"
    ].astype(int)

    validation_df["target"] = validation_df[
        "target"
    ].astype(int)

    valid_targets = {0, 1}

    if not set(train_df["target"].unique()).issubset(
        valid_targets
    ):
        raise ValueError(
            "The target column must contain only 0 and 1."
        )

    train_df = add_resolved_paths(
        train_df,
        image_column,
        PROJECT_ROOT,
        metadata_path,
        data_config,
    )

    validation_df = add_resolved_paths(
        validation_df,
        image_column,
        PROJECT_ROOT,
        metadata_path,
        data_config,
    )

    original_training_count = len(train_df)

    train_df = limit_training_images(
        train_df,
        args.max_images,
        seed,
    )

    print(
        f"Using {len(train_df)} of "
        f"{original_training_count} training images."
    )

    print(
        "Training class distribution:",
        train_df["target"].value_counts().to_dict(),
    )

    print(
        f"Using {len(validation_df)} validation images."
    )

    image_size = data_config["image_size"]
    batch_size = data_config["batch_size"]

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.RandomAffine(
                degrees=7.2,
                scale=(0.90, 1.10),
            ),
            transforms.ColorJitter(
                contrast=0.10
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = XRayDataset(
        train_df,
        transform=train_transform,
    )

    validation_dataset = XRayDataset(
        validation_df,
        transform=validation_transform,
    )

    num_workers = data_config.get(
        "num_workers",
        0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    # Calculate the equivalent of Keras class weights.
    negative_count = int(
        (train_df["target"] == 0).sum()
    )

    positive_count = int(
        (train_df["target"] == 1).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "The training sample contains no positive images."
        )

    pos_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"Positive loss weight: {pos_weight.item():.4f}"
    )

    results_dir = (
        PROJECT_ROOT
        / config["results"]["directory"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configured_model_path = (
        PROJECT_ROOT
        / config["model"]["path"]
    )

    # Convert a Keras filename to a PyTorch filename.
    if configured_model_path.suffix.lower() in [
        ".keras",
        ".h5",
    ]:
        model_path = configured_model_path.with_suffix(
            ".pth"
        )
    else:
        model_path = configured_model_path

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_weights = {
        0: len(train_df) / (2.0 * negative_count),
        1: len(train_df) / (2.0 * positive_count),
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

    model = build_model()
    model = model.to(device)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if batch_idx == 0:
            print("Outputs:", outputs.device, tuple(outputs.shape))
            print("Model:", next(model.parameters()).device)
            print("Images:", images.device)
            print("Labels:", labels.device)
            print("GPU memory allocated (MiB):",
                  round(torch.cuda.memory_allocated() / 1024 ** 2, 1))

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=config["training"]["learning_rate"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    epochs = config["training"]["epochs"]
    patience = 3
    best_auc = -float("inf")
    epochs_without_improvement = 0
    best_model_state = copy.deepcopy(
        model.state_dict()
    )

    history_records = []

    for epoch in range(epochs):
        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            optimizer=None,
            device=device,
            scaler=scaler,
        )

        scheduler.step(
            validation_metrics["roc_auc"]
        )

        print(
            f"Train loss:      {train_metrics['loss']:.4f}"
        )
        print(
            f"Train accuracy:  {train_metrics['accuracy']:.4f}"
        )
        print(
            f"Train precision: {train_metrics['precision']:.4f}"
        )
        print(
            f"Train recall:    {train_metrics['recall']:.4f}"
        )
        print(
            f"Train ROC-AUC:   {train_metrics['roc_auc']:.4f}"
        )

        print(
            f"Val loss:        {validation_metrics['loss']:.4f}"
        )
        print(
            f"Val accuracy:    {validation_metrics['accuracy']:.4f}"
        )
        print(
            f"Val precision:   {validation_metrics['precision']:.4f}"
        )
        print(
            f"Val recall:      {validation_metrics['recall']:.4f}"
        )
        print(
            f"Val ROC-AUC:     {validation_metrics['roc_auc']:.4f}"
        )
        print(
            f"Learning rate:   {optimizer.param_groups[0]['lr']:.7f}"
        )

        history_records.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_roc_auc": train_metrics["roc_auc"],
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
                "val_precision": validation_metrics["precision"],
                "val_recall": validation_metrics["recall"],
                "val_roc_auc": validation_metrics["roc_auc"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        current_auc = validation_metrics["roc_auc"]

        if current_auc > best_auc:
            best_auc = current_auc
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=validation_metrics,
                model_path=model_path,
                image_size=image_size,
            )

            print(
                f"Saved best model to: {model_path}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    # Restore the best initial model.
    model.load_state_dict(best_model_state)

    # Optional fine-tuning.
    if args.fine_tune:
        print("\nStarting fine-tuning...")

        prepare_for_fine_tuning(model)
        model = model.to(device)

        optimizer = torch.optim.Adam(
            filter(
                lambda parameter: parameter.requires_grad,
                model.parameters(),
            ),
            lr=config["training"][
                "fine_tune_learning_rate"
            ],
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda",
        )

        best_fine_tune_auc = -float("inf")
        fine_tune_epochs = config["training"][
            "fine_tune_epochs"
        ]

        for epoch in range(fine_tune_epochs):
            print(
                f"\nFine-tune epoch "
                f"{epoch + 1}/{fine_tune_epochs}"
            )

            train_metrics = run_epoch(
                model=model,
                data_loader=train_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
            )

            validation_metrics = run_epoch(
                model=model,
                data_loader=validation_loader,
                loss_function=loss_function,
                optimizer=None,
                device=device,
                scaler=scaler,
            )

            print(
                f"Train loss:    {train_metrics['loss']:.4f}"
            )
            print(
                f"Val loss:      {validation_metrics['loss']:.4f}"
            )
            print(
                f"Val accuracy:  {validation_metrics['accuracy']:.4f}"
            )
            print(
                f"Val ROC-AUC:   {validation_metrics['roc_auc']:.4f}"
            )

            if (
                validation_metrics["roc_auc"]
                > best_fine_tune_auc
            ):
                best_fine_tune_auc = (
                    validation_metrics["roc_auc"]
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    metrics=validation_metrics,
                    model_path=model_path,
                    image_size=image_size,
                )

                print(
                    f"Saved fine-tuned model to: {model_path}"
                )

    history_path = (
        results_dir / "training_history.csv"
    )

    pd.DataFrame(history_records).to_csv(
        history_path,
        index=False,
    )

    print(f"\nSaved history to: {history_path}")
    print(f"Saved model to: {model_path}")
    print(f"Training device: {device}")

    if device.type == "cuda":
        print(
            f"GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )


if __name__ == "__main__":
    main()