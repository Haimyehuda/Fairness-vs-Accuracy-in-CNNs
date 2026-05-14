"""
Augmentation Experiment (Pre-processing)

This script runs a controlled experiment using Data Augmentation
as a pre-processing fairness mitigation technique.

Research guarantees:
- DOES NOT modify the baseline run_experiment.py
- Uses the same SCENARIOS definition
- Uses the same LOCKED evaluation reference set
- Applies augmentation ONLY to the TRAIN set
- Logs results consistently to:
  - Persistent CSV (Google Drive)
  - Google Sheets (shared research ledger)

The experiment is fully reproducible and protocol-aligned.
"""

# ===============================================================
# Standard imports
# ===============================================================
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

# ===============================================================
# Path setup (robust)
# ===============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ===============================================================
# Project imports
# ===============================================================
from project.common.config import *
from project.scripts.scenarios import SCENARIOS
from project.common.dataset import XRTDataset
from project.common.model import get_model
from project.common.pipeline.train import train_model
from project.common.pipeline.eval import evaluate_model
from project.common.experiment_logger import log_experiment_to_sheets


# ===============================================================
# Argument parsing
# ===============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pre-processing augmentation experiment"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=SCENARIOS.keys(),
        help="Imbalance scenario identifier (e.g., 50-50, 10-90, 1-99)",
    )
    return parser.parse_args()


# ===============================================================
# Main experiment logic
# ===============================================================
def main():
    args = parse_args()
    scenario = SCENARIOS[args.scenario]

    # -----------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # -----------------------------------------------------------
    # Load evaluation reference (LOCKED)
    # -----------------------------------------------------------
    assert os.path.exists(
        EVAL_INDEX_PATH
    ), "Evaluation reference not found. Run Eval init first."

    eval_df = pd.read_csv(EVAL_INDEX_PATH)
    eval_paths = set(eval_df["image_path"])

    # -----------------------------------------------------------
    # Load CheXpert metadata
    # -----------------------------------------------------------
    df = pd.read_csv(os.path.join(CHEXPERT_ROOT, "train.csv"))
    df["label"] = np.where(df["Pneumonia"] == 1, POS_LABEL, NEG_LABEL)

    df["image_path"] = df["Path"].apply(
        lambda p: os.path.join(
            CHEXPERT_ROOT,
            p.replace("CheXpert-v1.0-small/", "").lstrip("/"),
        )
    )

    # -----------------------------------------------------------
    # Exclude eval samples from training pool
    # -----------------------------------------------------------
    train_pool = df[~df["image_path"].isin(eval_paths)]

    # -----------------------------------------------------------
    # Build TRAIN set (scenario-specific)
    # -----------------------------------------------------------
    rng = np.random.RandomState(SEED)

    train_pos = train_pool[train_pool["label"] == POS_LABEL].sample(
        n=scenario["n_pneumonia"], random_state=rng
    )

    train_neg = train_pool[train_pool["label"] == NEG_LABEL].sample(
        n=scenario["n_normal"], random_state=rng
    )

    train_df = pd.concat([train_pos, train_neg], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    print(f"\nRunning AUGMENTATION experiment: {scenario['name']}")
    print("TRAIN SET")
    print("  PNEUMONIA:", (train_df["label"] == POS_LABEL).sum())
    print("  NORMAL   :", (train_df["label"] == NEG_LABEL).sum())
    print("  TOTAL    :", len(train_df))

    # -----------------------------------------------------------
    # Transforms
    # -----------------------------------------------------------
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    # -----------------------------------------------------------
    # Dataset & loaders
    # -----------------------------------------------------------
    train_ds = XRTDataset(train_df, transform=train_transform)
    eval_ds = XRTDataset(eval_df, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------------
    # Model training
    # -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
    )

    # -----------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------
    plots_dir = os.path.join(DRIVE_ROOT, scenario["name"] + "_AUG")

    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
        make_plots=True,
        plots_dir=plots_dir,
        run_name=scenario["name"] + "_AUG",
        research_title=RESEARCH_TITLE,
    )

    # -----------------------------------------------------------
    # Build results row (schema MUST match baseline)
    # -----------------------------------------------------------
    row = {
        "Method": "Data Augmentation",
        "Scenario": args.scenario,
        "Train Ratio (P/N)": f"{scenario['n_pneumonia']}/{scenario['n_normal']}",
        "#P Train": scenario["n_pneumonia"],
        "#N Train": scenario["n_normal"],
        "#Train After Processing": len(train_df),
        "Accuracy (Overall)": metrics["accuracy"],
        "Accuracy NORMAL": metrics["acc_normal"],
        "Accuracy PNEUMONIA": metrics["acc_pneumonia"],
        "Recall / TPR NORMAL": metrics["tpr_normal"],
        "Recall / TPR PNEUMONIA": metrics["tpr_pneumonia"],
        "F1 NORMAL": metrics["f1_normal"],
        "F1 PNEUMONIA": metrics["f1_pneumonia"],
        "ΔTPR (Equal Opportunity)": metrics["delta_tpr"],
        "ΔFPR (Equalized Odds)": metrics["delta_fpr"],
        "Disparate Impact (DI)": metrics["disparate_impact"],
    }

    # -----------------------------------------------------------
    # Persist results to CSV (Drive)
    # -----------------------------------------------------------
    df_row = pd.DataFrame([row])

    # Ensure unified column order
    df_row = df_row.reindex(columns=RESULT_COLUMNS)

    if os.path.exists(RESULTS_PATH):
        try:
            existing_df = pd.read_csv(RESULTS_PATH)
            updated_df = pd.concat([existing_df, df_row], ignore_index=True)
            updated_df = updated_df.reindex(columns=RESULT_COLUMNS)
            updated_df.to_csv(RESULTS_PATH, index=False)
        except Exception:
            df_row.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(RESULTS_PATH, index=False)

    # -----------------------------------------------------------
    # Persist results to Google Sheets
    # -----------------------------------------------------------
    try:
        log_experiment_to_sheets(row, EXPERIMENT_SHEET_ID)
        print("✔ Results appended to Google Sheets")
    except Exception as e:
        print("⚠️ Failed to write to Google Sheets:", e)

    # -----------------------------------------------------------
    # Final confirmation
    # -----------------------------------------------------------
    print("\n✔ Augmentation experiment completed successfully")
    print(df_row)


if __name__ == "__main__":
    main()
