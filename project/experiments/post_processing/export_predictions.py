# ===============================================================
# POST-PROCESSING STEP 1 — EXPORT PREDICTIONS
# File:
#   project/experiments/post_processing/export_predictions.py
#
# Purpose:
#   - Load a trained checkpoint dynamically by method + scenario
#   - Run the model on the fixed locked Eval set
#   - Export per-image probabilities for Post-processing
#
# Output:
#   /content/drive/MyDrive/Fairness-vs-Accuracy-in-CNNs/predictions/
#       {METHOD}_{SCENARIO}_predictions.csv
#
# Example:
#   python experiments/post_processing/export_predictions.py \
#     --method "Logit Adjustment" \
#     --scenario 1-99
# ===============================================================

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ===============================================================
# Robust sys.path setup
# ===============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# export_predictions.py is located at:
# project/experiments/post_processing/export_predictions.py
PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..")
)

COMMON_PATH = os.path.join(PROJECT_ROOT, "common")

if COMMON_PATH not in sys.path:
    sys.path.insert(0, COMMON_PATH)


from dataset import XRTDataset
from model import get_model
from utils import get_device


# ===============================================================
# Constants
# ===============================================================

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Fairness-vs-Accuracy-in-CNNs"

DEFAULT_EVAL_INDEX_PATH = os.path.join(
    DEFAULT_DRIVE_ROOT,
    "eval_reference",
    "eval_index.csv",
)

DEFAULT_CHECKPOINT_DIR = os.path.join(
    DEFAULT_DRIVE_ROOT,
    "checkpoints",
)

DEFAULT_OUTPUT_DIR = os.path.join(
    DEFAULT_DRIVE_ROOT,
    "predictions",
)

LABEL_TO_ID = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}

ID_TO_LABEL = {
    0: "NORMAL",
    1: "PNEUMONIA",
}


# ===============================================================
# Helpers
# ===============================================================

def safe_name(text: str) -> str:
    """
    Converts method/scenario names into stable filenames.

    Examples:
        "Focal Loss" -> "Focal_Loss"
        "Oversampling + Augmentation" -> "Oversampling_plus_Augmentation"
        "1-99" -> "1-99"
    """
    return (
        str(text)
        .strip()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Eval predictions for post-processing."
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help='Method name, e.g. "Baseline", "Focal Loss", "Logit Adjustment"',
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help='Scenario name, e.g. "50-50", "60-40", "10-90", "1-99"',
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional explicit checkpoint path. "
            "If not provided, path is built dynamically from method + scenario."
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing checkpoint .pth files.",
    )

    parser.add_argument(
        "--eval-index",
        type=str,
        default=DEFAULT_EVAL_INDEX_PATH,
        help="Path to locked eval_index.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where prediction CSV files will be saved.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for Eval prediction export.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )

    return parser.parse_args()


def resolve_checkpoint_path(args, method_safe: str, scenario_safe: str) -> str:
    """
    If --checkpoint is provided, use it.
    Otherwise build:
        {checkpoint_dir}/{method_safe}_{scenario_safe}.pth
    """
    if args.checkpoint:
        return args.checkpoint

    return os.path.join(
        args.checkpoint_dir,
        f"{method_safe}_{scenario_safe}.pth",
    )


def validate_paths(eval_index_path: str, checkpoint_path: str):
    if not os.path.exists(eval_index_path):
        raise FileNotFoundError(
            f"Eval index not found:\n{eval_index_path}"
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}\n\n"
            "Expected checkpoint naming format:\n"
            "{METHOD}_{SCENARIO}.pth\n\n"
            "Examples:\n"
            "Baseline_1-99.pth\n"
            "Data_Augmentation_1-99.pth\n"
            "Oversampling_plus_Augmentation_1-99.pth\n"
            "Class_Weighting_1-99.pth\n"
            "Focal_Loss_1-99.pth\n"
            "Logit_Adjustment_1-99.pth"
        )


def validate_eval_df(eval_df: pd.DataFrame):
    required_columns = ["image_path", "label"]

    for col in required_columns:
        if col not in eval_df.columns:
            raise ValueError(
                f"eval_index.csv must contain column: {col}"
            )

    unknown_labels = set(eval_df["label"].unique()) - set(LABEL_TO_ID.keys())

    if unknown_labels:
        raise ValueError(
            f"Unknown labels in eval_index.csv: {unknown_labels}. "
            f"Allowed labels: {list(LABEL_TO_ID.keys())}"
        )


def load_checkpoint(model, checkpoint_path: str, device):
    """
    Supports common checkpoint formats:
    1. Raw state_dict
    2. {"model_state_dict": ...}
    3. {"state_dict": ...}
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def extract_batch(batch, eval_df: pd.DataFrame, row_index: int):
    """
    Supports Dataset outputs:
    1. (images, labels)
    2. (images, labels, paths)

    If paths are not returned by Dataset,
    paths are taken from eval_df according to DataLoader order.
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        images, labels, paths = batch

    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, labels = batch
        paths = eval_df.iloc[
            row_index : row_index + len(images)
        ]["image_path"].tolist()

    else:
        raise ValueError(
            "Unsupported batch format. Expected: "
            "(images, labels) or (images, labels, paths)"
        )

    return images, labels, paths


def labels_to_numpy(labels):
    """
    Converts labels to numpy array.
    Handles tensor labels and basic python/numpy labels.
    """
    if torch.is_tensor(labels):
        return labels.detach().cpu().numpy()

    return np.array(labels)


def normalize_label_ids(labels_np):
    """
    Ensures labels are numeric IDs: NORMAL=0, PNEUMONIA=1.

    If Dataset already returns integers, keep them.
    If Dataset returns strings, map them.
    """
    normalized = []

    for value in labels_np:
        if isinstance(value, str):
            normalized.append(LABEL_TO_ID[value])
        else:
            normalized.append(int(value))

    return np.array(normalized, dtype=int)


# ===============================================================
# Main
# ===============================================================

def main():
    args = parse_args()

    method_safe = safe_name(args.method)
    scenario_safe = safe_name(args.scenario)

    checkpoint_path = resolve_checkpoint_path(
        args=args,
        method_safe=method_safe,
        scenario_safe=scenario_safe,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(
        args.output_dir,
        f"{method_safe}_{scenario_safe}_predictions.csv",
    )

    print("======================================")
    print(" POST-PROCESSING STEP 1")
    print(" EXPORT EVAL PREDICTIONS")
    print("======================================")
    print("Method         :", args.method)
    print("Scenario       :", args.scenario)
    print("Eval index     :", args.eval_index)
    print("Checkpoint     :", checkpoint_path)
    print("Output path    :", output_path)
    print("Batch size     :", args.batch_size)
    print("Num workers    :", args.num_workers)

    validate_paths(
        eval_index_path=args.eval_index,
        checkpoint_path=checkpoint_path,
    )

    # -----------------------------------------------------------
    # Load locked Eval
    # -----------------------------------------------------------
    eval_df = pd.read_csv(args.eval_index)
    validate_eval_df(eval_df)

    print("\nEval rows:", len(eval_df))
    print("Eval label counts:")
    print(eval_df["label"].value_counts())

    # -----------------------------------------------------------
    # Dataset / DataLoader
    # -----------------------------------------------------------
    eval_dataset = XRTDataset(eval_df, transform=eval_transform)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # -----------------------------------------------------------
    # Model
    # -----------------------------------------------------------
    device = get_device()
    print("\nDevice:", device)

    model = get_model(num_classes=2)
    model = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model = model.to(device)
    model.eval()

    print("✔ Model loaded successfully")

    # -----------------------------------------------------------
    # Export predictions
    # -----------------------------------------------------------
    rows = []
    row_index = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Running locked Eval"):
            images, labels, paths = extract_batch(
                batch=batch,
                eval_df=eval_df,
                row_index=row_index,
            )

            images = images.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            labels_np = labels_to_numpy(labels)
            labels_np = normalize_label_ids(labels_np)

            for i in range(len(images)):
                y_true_id = int(labels_np[i])
                y_pred_id = int(preds[i])

                rows.append({
                    "method": args.method,
                    "scenario": args.scenario,
                    "image_path": paths[i],
                    "y_true": ID_TO_LABEL[y_true_id],
                    "y_true_id": y_true_id,
                    "prob_NORMAL": float(probs[i][0]),
                    "prob_PNEUMONIA": float(probs[i][1]),
                    "y_pred_original": ID_TO_LABEL[y_pred_id],
                    "y_pred_original_id": y_pred_id,
                })

            row_index += len(images)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_path, index=False)

    print("\n✔ Predictions exported successfully")
    print("Rows saved:", len(pred_df))
    print("Saved to  :", output_path)

    print("\nOriginal prediction counts:")
    print(pred_df["y_pred_original"].value_counts())

    print("\nPreview:")
    print(pred_df.head())


if __name__ == "__main__":
    main()