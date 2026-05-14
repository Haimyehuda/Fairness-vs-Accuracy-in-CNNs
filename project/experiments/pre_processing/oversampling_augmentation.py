"""
Oversampling + Augmentation Experiment (Pre-processing)

Applies:
1. Oversampling to balance the TRAIN set
2. Data Augmentation only on the TRAIN set

Eval set remains fixed and unchanged.
"""

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run oversampling + augmentation experiment"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=SCENARIOS.keys(),
    )
    return parser.parse_args()


def oversample_minority(train_df: pd.DataFrame) -> pd.DataFrame:
    counts = train_df["label"].value_counts()

    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    majority_df = train_df[train_df["label"] == majority_label]
    minority_df = train_df[train_df["label"] == minority_label]

    minority_upsampled = minority_df.sample(
        n=len(majority_df),
        replace=True,
        random_state=SEED,
    )

    balanced_df = pd.concat(
        [majority_df, minority_upsampled],
        ignore_index=True,
    )

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    print("\nOVERSAMPLING")
    print("Before:", counts.to_dict())
    print("After :", balanced_df["label"].value_counts().to_dict())

    return balanced_df


def main():
    args = parse_args()
    scenario = SCENARIOS[args.scenario]

    run_name = scenario["name"] + "_OS_AUG"

    print(f"\n=== {RESEARCH_TITLE} ===")
    print(f"Running OVERSAMPLING + AUGMENTATION experiment: {run_name}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    assert os.path.exists(EVAL_INDEX_PATH), "Evaluation reference not found."

    eval_df = pd.read_csv(EVAL_INDEX_PATH)
    eval_paths = set(eval_df["image_path"])

    df = pd.read_csv(os.path.join(CHEXPERT_ROOT, "train.csv"))
    df["label"] = np.where(df["Pneumonia"] == 1, POS_LABEL, NEG_LABEL)

    df["image_path"] = df["Path"].apply(
        lambda p: os.path.join(
            CHEXPERT_ROOT,
            p.replace("CheXpert-v1.0-small/", "").lstrip("/"),
        )
    )

    train_pool = df[~df["image_path"].isin(eval_paths)]

    rng = np.random.RandomState(SEED)

    train_pos = train_pool[train_pool["label"] == POS_LABEL].sample(
        n=scenario["n_pneumonia"],
        random_state=rng,
    )

    train_neg = train_pool[train_pool["label"] == NEG_LABEL].sample(
        n=scenario["n_normal"],
        random_state=rng,
    )

    train_df_original = pd.concat(
        [train_pos, train_neg],
        ignore_index=True,
    )

    train_df_original = train_df_original.sample(
        frac=1,
        random_state=rng,
    ).reset_index(drop=True)

    train_df = oversample_minority(train_df_original)

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

    train_ds = XRTDataset(train_df, transform=train_transform)
    eval_ds = XRTDataset(eval_df, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
    )

    plots_dir = os.path.join(DRIVE_ROOT, run_name)

    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
        make_plots=True,
        plots_dir=plots_dir,
        run_name=run_name,
        research_title=RESEARCH_TITLE,
    )

    row = {
        "Method": "Oversampling + Augmentation",
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

    try:
        log_experiment_to_sheets(row, EXPERIMENT_SHEET_ID)
        print("✔ Results appended to Google Sheets")
    except Exception as e:
        print("⚠️ Failed to write to Google Sheets:", e)

    print("\n✔ Oversampling + Augmentation experiment completed successfully")
    print(df_row)


if __name__ == "__main__":
    main()