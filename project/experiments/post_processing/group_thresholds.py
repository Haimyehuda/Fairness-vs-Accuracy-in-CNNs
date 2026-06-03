# ===============================================================
# POST-PROCESSING — GROUP-SPECIFIC / CLASS-SPECIFIC THRESHOLDS
# File:
#   project/experiments/post_processing/group_thresholds.py
#
# Purpose:
#   - Implement Group-Specific (Class-Specific) Thresholds as a
#     post-processing step.
#   - Reads existing prediction CSV files created by export_predictions.py.
#   - Does NOT train any model.
#   - Does NOT load PyTorch checkpoints.
#   - Does NOT write to Google Sheets.
#
# Run with:
#   python -m project.experiments.post_processing.group_thresholds \
#     --method "Baseline" \
#     --scenario 10-90
# ===============================================================

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

# ===============================================================
# Robust sys.path setup
# ===============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SCRIPT_DIR is project/experiments/post_processing/
# PROJECT_ROOT is project/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
# REPO_ROOT is the parent directory containing project/
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Try importing SCENARIOS from scripts
try:
    from project.scripts.scenarios import SCENARIOS
except ImportError:
    SCENARIOS = {}

# ===============================================================
# Constants and Configurations
# ===============================================================

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Fairness-vs-Accuracy-in-CNNs"

# Safe names mapping (as defined in export_predictions.py)
SAFE_METHODS = {
    "Baseline": "Baseline",
    "Data Augmentation": "Data_Augmentation",
    "Oversampling": "Oversampling",
    "Oversampling + Augmentation": "Oversampling_plus_Augmentation",
    "Class Weighting": "Class_Weighting",
    "Focal Loss": "Focal_Loss",
    "Logit Adjustment": "Logit_Adjustment",
}

# Column schemas
BEST_RESULT_COLUMNS = [
    "Method",
    "Base Method",
    "Scenario",
    "Selected NORMAL Threshold",
    "Selected PNEUMONIA Threshold",
    "Train Ratio (P/N)",
    "#P Train",
    "#N Train",
    "#Train After Processing",
    "Accuracy (Overall)",
    "Accuracy NORMAL",
    "Accuracy PNEUMONIA",
    "Recall / TPR NORMAL",
    "Recall / TPR PNEUMONIA",
    "F1 NORMAL",
    "F1 PNEUMONIA",
    "ΔTPR (Equal Opportunity)",
    "ΔFPR (Equalized Odds)",
    "Disparate Impact (DI)",
]

SWEEP_COLUMNS = [
    "Normal Threshold",
    "Pneumonia Threshold",
    "Accuracy (Overall)",
    "Accuracy NORMAL",
    "Accuracy PNEUMONIA",
    "Recall / TPR NORMAL",
    "Recall / TPR PNEUMONIA",
    "F1 NORMAL",
    "F1 PNEUMONIA",
    "ΔTPR (Equal Opportunity)",
    "ΔFPR (Equalized Odds)",
    "Disparate Impact (DI)",
]


# ===============================================================
# Helpers
# ===============================================================

def safe_name(text: str) -> str:
    """
    Converts method/scenario names into stable filenames.
    Must match safe_name in export_predictions.py.
    """
    cleaned = text.strip()
    if cleaned in SAFE_METHODS:
        return SAFE_METHODS[cleaned]

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates accuracy, recall (TPR), F1, and fairness gaps (ΔTPR, ΔFPR, DI).
    Ensures mathematical definitions strictly align with project/common/pipeline/eval.py
    and threshold_calibration.py.
    """
    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # Accuracy
    accuracy = float((y_pred == y_true).mean())

    # Per-class accuracy
    acc_normal = (
        float((y_pred[y_true == 0] == 0).mean()) if (y_true == 0).any() else 0.0
    )
    acc_pneumonia = (
        float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else 0.0
    )

    # TPR / FPR (by class)
    tpr_normal = float(TN / (TN + FP + 1e-12))
    tpr_pneumonia = float(TP / (TP + FN + 1e-12))

    fpr_normal = float(FN / (TP + FN + 1e-12))
    fpr_pneumonia = float(FP / (TN + FP + 1e-12))

    # Fairness gaps
    delta_tpr = float(abs(tpr_normal - tpr_pneumonia))
    delta_fpr = float(abs(fpr_normal - fpr_pneumonia))

    # Disparate Impact (positive prediction rate ratio)
    ppr_normal = (
        float((y_pred[y_true == 0] == 1).mean()) if (y_true == 0).any() else 0.0
    )
    ppr_pneumonia = (
        float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else 0.0
    )
    disparate_impact = (
        float(ppr_pneumonia / (ppr_normal + 1e-12)) if ppr_normal > 0 else 0.0
    )

    # F1-score per class (computed over the full dataset)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    f1_normal = float(f1_per_class[0])
    f1_pneumonia = float(f1_per_class[1])

    return {
        "Accuracy (Overall)": accuracy,
        "Accuracy NORMAL": acc_normal,
        "Accuracy PNEUMONIA": acc_pneumonia,
        "Recall / TPR NORMAL": tpr_normal,
        "Recall / TPR PNEUMONIA": tpr_pneumonia,
        "F1 NORMAL": f1_normal,
        "F1 PNEUMONIA": f1_pneumonia,
        "ΔTPR (Equal Opportunity)": delta_tpr,
        "ΔFPR (Equalized Odds)": delta_fpr,
        "Disparate Impact (DI)": disparate_impact,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run post-processing Group-Specific / Class-Specific Thresholds."
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
        "--drive-root",
        type=str,
        default=DEFAULT_DRIVE_ROOT,
        help="Google Drive root directory path.",
    )

    return parser.parse_args()


# ===============================================================
# Main Logic
# ===============================================================

def main():
    args = parse_args()

    # Determine paths
    drive_root = args.drive_root
    predictions_dir = os.path.join(drive_root, "predictions")
    sweeps_dir = os.path.join(drive_root, "post_processing", "group_specific_thresholds", "sweeps")
    results_dir = os.path.join(drive_root, "post_processing", "group_specific_thresholds", "results")

    method_safe = safe_name(args.method)
    scenario_safe = safe_name(args.scenario)

    pred_filename = f"{method_safe}_{scenario_safe}_predictions.csv"
    pred_path = os.path.join(predictions_dir, pred_filename)

    if not os.path.exists(pred_path):
        print(f"Error: Predictions file not found at:\n  {pred_path}", file=sys.stderr)
        print("Please run export_predictions.py first for this method and scenario.", file=sys.stderr)
        sys.exit(1)

    print("=============================================================")
    print(" POST-PROCESSING — GROUP-SPECIFIC / CLASS-SPECIFIC THRESHOLDS")
    print("=============================================================")
    print(f"Method:            {args.method}")
    print(f"Scenario:          {args.scenario}")
    print(f"Predictions file:  {pred_path}")

    # Read predictions
    df = pd.read_csv(pred_path)

    # Check for expected columns
    required_cols = ["y_true_id", "prob_NORMAL", "prob_PNEUMONIA"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' missing from predictions CSV.", file=sys.stderr)
            sys.exit(1)

    y_true = df["y_true_id"].to_numpy().astype(int)
    prob_normal = df["prob_NORMAL"].to_numpy().astype(float)
    prob_pneumonia = df["prob_PNEUMONIA"].to_numpy().astype(float)

    # Generate thresholds from 0.05 to 0.95 (steps of 0.01)
    thresholds = [round(0.05 + i * 0.01, 2) for i in range(91)]

    print("Running grid search over 8,281 threshold pairs...")
    sweep_results = []
    
    for t_normal in thresholds:
        for t_pneumonia in thresholds:
            # Group/Class-Specific thresholding scores:
            normal_score = prob_normal / t_normal
            pneumonia_score = prob_pneumonia / t_pneumonia

            # Prediction rule: if normal_score >= pneumonia_score => NORMAL (0) else PNEUMONIA (1)
            y_pred = np.where(normal_score >= pneumonia_score, 0, 1)

            metrics = compute_metrics(y_true, y_pred)
            metrics["Normal Threshold"] = t_normal
            metrics["Pneumonia Threshold"] = t_pneumonia
            sweep_results.append(metrics)

    sweep_df = pd.DataFrame(sweep_results)

    # Calculate distance to (0.50, 0.50) to use in tertiary selection logic
    sweep_df["dist_to_0_5"] = (
        (sweep_df["Normal Threshold"] - 0.50) ** 2 + 
        (sweep_df["Pneumonia Threshold"] - 0.50) ** 2
    ).pow(0.5).round(6)

    # Selection logic:
    # 1. Minimize ΔTPR (Equal Opportunity)
    # 2. Maximize Accuracy (Overall)
    # 3. Choose threshold pair closest to (0.50, 0.50)
    sorted_df = sweep_df.sort_values(
        by=["ΔTPR (Equal Opportunity)", "Accuracy (Overall)", "dist_to_0_5"],
        ascending=[True, False, True]
    )

    best_row = sorted_df.iloc[0]
    selected_normal_threshold = float(best_row["Normal Threshold"])
    selected_pneumonia_threshold = float(best_row["Pneumonia Threshold"])

    # Extract original threshold 0.50 / 0.50 behavior for comparison
    row_050 = sweep_df[
        np.isclose(sweep_df["Normal Threshold"], 0.50) & 
        np.isclose(sweep_df["Pneumonia Threshold"], 0.50)
    ].iloc[0]

    # Resolve training set statistics from SCENARIOS if available
    train_ratio = ""
    n_p_train = ""
    n_n_train = ""
    n_train_processed = ""

    if args.scenario in SCENARIOS:
        scen = SCENARIOS[args.scenario]
        p_count = scen.get("n_pneumonia")
        n_count = scen.get("n_normal")
        if p_count is not None and n_count is not None:
            train_ratio = f"{p_count}/{n_count}"
            n_p_train = p_count
            n_n_train = n_count

            # Determine processed size based on method
            method_clean = args.method.strip()
            if method_clean in ["Oversampling", "Oversampling + Augmentation", "Oversampling_plus_Augmentation"]:
                n_train_processed = 2 * max(p_count, n_count)
            else:
                n_train_processed = p_count + n_count

    # Prepare outputs
    os.makedirs(sweeps_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save sweep results
    sweep_path = os.path.join(sweeps_dir, f"{method_safe}_{scenario_safe}_group_thresholds_sweep.csv")
    sweep_df_save = sweep_df[SWEEP_COLUMNS]
    sweep_df_save.to_csv(sweep_path, index=False)

    # Save selected best result
    result_path = os.path.join(results_dir, f"{method_safe}_{scenario_safe}_group_thresholds_result.csv")
    best_result = {
        "Method": "Group-Specific Thresholds",
        "Base Method": args.method,
        "Scenario": args.scenario,
        "Selected NORMAL Threshold": selected_normal_threshold,
        "Selected PNEUMONIA Threshold": selected_pneumonia_threshold,
        "Train Ratio (P/N)": train_ratio,
        "#P Train": n_p_train,
        "#N Train": n_n_train,
        "#Train After Processing": n_train_processed,
        "Accuracy (Overall)": best_row["Accuracy (Overall)"],
        "Accuracy NORMAL": best_row["Accuracy NORMAL"],
        "Accuracy PNEUMONIA": best_row["Accuracy PNEUMONIA"],
        "Recall / TPR NORMAL": best_row["Recall / TPR NORMAL"],
        "Recall / TPR PNEUMONIA": best_row["Recall / TPR PNEUMONIA"],
        "F1 NORMAL": best_row["F1 NORMAL"],
        "F1 PNEUMONIA": best_row["F1 PNEUMONIA"],
        "ΔTPR (Equal Opportunity)": best_row["ΔTPR (Equal Opportunity)"],
        "ΔFPR (Equalized Odds)": best_row["ΔFPR (Equalized Odds)"],
        "Disparate Impact (DI)": best_row["Disparate Impact (DI)"],
    }

    best_df = pd.DataFrame([best_result])[BEST_RESULT_COLUMNS]
    best_df.to_csv(result_path, index=False)

    # Print summary comparison
    print("\n=============================================================")
    print(" GROUP-SPECIFIC / CLASS-SPECIFIC THRESHOLD SUMMARY COMPARISON")
    print("=============================================================")
    print(f"Method:                 {args.method}")
    print(f"Scenario:               {args.scenario}")
    print(f"Prediction File:        {pred_path}")
    print(f"Selected Thresholds:    NORMAL={selected_normal_threshold:.2f}, PNEUMONIA={selected_pneumonia_threshold:.2f}")
    print("-------------------------------------------------------------")
    print(f"{'Metric':<30} | {'Thresholds 0.50/0.50 (Orig)':<27} | {f'Selected {selected_normal_threshold:.2f}/{selected_pneumonia_threshold:.2f}':<25}")
    print("-" * 89)

    print(f"{'Accuracy (Overall)':<30} | {row_050['Accuracy (Overall)']:<27.4f} | {best_row['Accuracy (Overall)']:<25.4f}")
    print(f"{'Accuracy NORMAL':<30} | {row_050['Accuracy NORMAL']:<27.4f} | {best_row['Accuracy NORMAL']:<25.4f}")
    print(f"{'Accuracy PNEUMONIA':<30} | {row_050['Accuracy PNEUMONIA']:<27.4f} | {best_row['Accuracy PNEUMONIA']:<25.4f}")
    print(f"{'Recall / TPR NORMAL':<30} | {row_050['Recall / TPR NORMAL']:<27.4f} | {best_row['Recall / TPR NORMAL']:<25.4f}")
    print(f"{'Recall / TPR PNEUMONIA':<30} | {row_050['Recall / TPR PNEUMONIA']:<27.4f} | {best_row['Recall / TPR PNEUMONIA']:<25.4f}")
    print(f"{'F1 NORMAL':<30} | {row_050['F1 NORMAL']:<27.4f} | {best_row['F1 NORMAL']:<25.4f}")
    print(f"{'F1 PNEUMONIA':<30} | {row_050['F1 PNEUMONIA']:<27.4f} | {best_row['F1 PNEUMONIA']:<25.4f}")
    print(f"{'ΔTPR (Equal Opportunity)':<30} | {row_050['ΔTPR (Equal Opportunity)']:<27.4f} | {best_row['ΔTPR (Equal Opportunity)']:<25.4f}")
    print(f"{'ΔFPR (Equalized Odds)':<30} | {row_050['ΔFPR (Equalized Odds)']:<27.4f} | {best_row['ΔFPR (Equalized Odds)']:<25.4f}")
    print(f"{'Disparate Impact (DI)':<30} | {row_050['Disparate Impact (DI)']:<27.4f} | {best_row['Disparate Impact (DI)']:<25.4f}")

    print("-------------------------------------------------------------")
    print("Output Files Saved:")
    print(f"  Sweep results:      {sweep_path}")
    print(f"  Best result:        {result_path}")
    print("=============================================================")


if __name__ == "__main__":
    main()
