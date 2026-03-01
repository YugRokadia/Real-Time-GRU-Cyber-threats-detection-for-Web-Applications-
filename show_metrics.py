"""
show_metrics.py — Visualise all key metrics for the trained GRU model.

Reads:
  - training_history.json  (saved by train_model.py)
  - epoch_metrics.csv      (saved by CSVLogger in train_model.py)
  - test_results.npz       (saved by train_model.py)

Produces:
  - Confusion matrix heatmap
  - ROC curve with AUC
  - Precision-Recall curve with AP
  - Training / Validation loss & accuracy curves
  - Per-epoch metrics table from CSV
  - Per-class precision, recall, F1 bar chart
  - All figures saved as PNGs in a `metrics/` folder
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
)

OUTPUT_DIR = "metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading training history and test results …")

with open("training_history.json", "r") as f:
    history = json.load(f)

# Load per-epoch CSV if available
epoch_csv_path = "epoch_metrics.csv"
epoch_df = None
if os.path.exists(epoch_csv_path):
    epoch_df = pd.read_csv(epoch_csv_path)
    print(f"Loaded per-epoch metrics from {epoch_csv_path} ({len(epoch_df)} epochs)")

data = np.load("test_results.npz")
y_test = data["y_test"]
y_pred = data["y_pred"]
y_pred_prob = data["y_pred_prob"]

CLASS_NAMES = ["Benign (0)", "Malicious (1)"]

# ---------------------------------------------------------------------------
# 2. Training / Validation curves
# ---------------------------------------------------------------------------
def plot_training_curves():
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    axes[0].plot(epochs, history["loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Annotate gap at last epoch
    gap = history["loss"][-1] - history["val_loss"][-1]
    axes[0].annotate(
        f"Gap: {abs(gap):.4f}",
        xy=(epochs[-1], history["val_loss"][-1]),
        fontsize=9,
        color="gray",
    )

    # --- Accuracy ---
    axes[1].plot(epochs, history["accuracy"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_accuracy"], "r-o", label="Val Acc", markersize=4)
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # --- Overfitting diagnostic ---
    best_epoch = int(np.argmin(history["val_loss"]))
    train_loss_at_best = history["loss"][best_epoch]
    val_loss_at_best = history["val_loss"][best_epoch]
    overfit_gap = train_loss_at_best - val_loss_at_best
    print(f"\n  Best epoch (lowest val_loss): {best_epoch + 1}")
    print(f"  Train loss: {train_loss_at_best:.4f}  |  Val loss: {val_loss_at_best:.4f}  |  Gap: {abs(overfit_gap):.4f}")
    if abs(overfit_gap) > 0.10:
        print("  ⚠️  Significant train/val gap — possible residual overfitting.")
    else:
        print("  ✅ Train/val gap is small — model generalises well.")


# ---------------------------------------------------------------------------
# 2b. Per-epoch metrics table & learning rate plot (from CSV)
# ---------------------------------------------------------------------------
def plot_epoch_csv_metrics():
    if epoch_df is None:
        print("  ⚠️ No epoch_metrics.csv found — skipping CSV-based plots.")
        return
    
    # --- Print tabular summary ---
    print("\n  Per-epoch metrics (from epoch_metrics.csv):")
    print("  " + "-" * 80)
    header = f"  {'Epoch':>5}  {'Loss':>10}  {'Acc':>10}  {'Val Loss':>10}  {'Val Acc':>10}  {'LR':>12}"
    print(header)
    print("  " + "-" * 80)
    for _, row in epoch_df.iterrows():
        epoch_num = int(row['epoch']) + 1
        lr_val = row.get('learning_rate', row.get('lr', 0))
        print(f"  {epoch_num:>5}  {row['loss']:>10.6f}  {row['accuracy']:>10.6f}  {row['val_loss']:>10.6f}  {row['val_accuracy']:>10.6f}  {lr_val:>12.8f}")
    print("  " + "-" * 80)
    
    # --- Save tabular summary to text file ---
    table_path = os.path.join(OUTPUT_DIR, "epoch_metrics_table.txt")
    with open(table_path, 'w') as f:
        f.write(f"{'Epoch':>5}  {'Loss':>10}  {'Accuracy':>10}  {'Val Loss':>10}  {'Val Acc':>10}  {'LR':>12}\n")
        f.write("-" * 75 + "\n")
        for _, row in epoch_df.iterrows():
            epoch_num = int(row['epoch']) + 1
            lr_val = row.get('learning_rate', row.get('lr', 0))
            f.write(f"{epoch_num:>5}  {row['loss']:>10.6f}  {row['accuracy']:>10.6f}  {row['val_loss']:>10.6f}  {row['val_accuracy']:>10.6f}  {lr_val:>12.8f}\n")
    print(f"  Saved {table_path}")
    
    # --- Learning rate schedule plot ---
    lr_col = 'learning_rate' if 'learning_rate' in epoch_df.columns else 'lr'
    if lr_col in epoch_df.columns:
        epochs = epoch_df['epoch'] + 1
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, epoch_df[lr_col], 'g-o', markersize=4, label='Learning Rate')
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "learning_rate.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")
    
    # --- Combined 4-panel plot from CSV ---
    epochs = epoch_df['epoch'] + 1
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, epoch_df['loss'], 'b-o', label='Train Loss', markersize=4)
    axes[0, 0].plot(epochs, epoch_df['val_loss'], 'r-o', label='Val Loss', markersize=4)
    axes[0, 0].set_title('Loss vs Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, epoch_df['accuracy'], 'b-o', label='Train Acc', markersize=4)
    axes[0, 1].plot(epochs, epoch_df['val_accuracy'], 'r-o', label='Val Acc', markersize=4)
    axes[0, 1].set_title('Accuracy vs Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Train-Val gap
    loss_gap = np.array(epoch_df['loss']) - np.array(epoch_df['val_loss'])
    axes[1, 0].plot(epochs, np.abs(loss_gap), 'm-o', markersize=4, label='|Train - Val| Loss')
    axes[1, 0].set_title('Overfitting Gap (Loss)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('|Gap|')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    lr_col = 'learning_rate' if 'learning_rate' in epoch_df.columns else 'lr'
    if lr_col in epoch_df.columns:
        axes[1, 1].plot(epochs, epoch_df[lr_col], 'g-o', markersize=4, label='LR')
        axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics Dashboard (from epoch_metrics.csv)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_dashboard.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 3. Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalised (recall-wise)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Oranges", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalised by row)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 4. ROC Curve
# ---------------------------------------------------------------------------
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}  |  AUC = {roc_auc:.4f}")


# ---------------------------------------------------------------------------
# 5. Precision-Recall Curve
# ---------------------------------------------------------------------------
def plot_pr_curve():
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    ap = average_precision_score(y_test, y_pred_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="purple", lw=2, label=f"PR (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}  |  AP = {ap:.4f}")


# ---------------------------------------------------------------------------
# 6. Per-class metrics bar chart
# ---------------------------------------------------------------------------
def plot_class_metrics():
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    metrics = ["precision", "recall", "f1-score"]
    benign_vals = [report[CLASS_NAMES[0]][m] for m in metrics]
    malicious_vals = [report[CLASS_NAMES[1]][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, benign_vals, width, label=CLASS_NAMES[0], color="steelblue")
    bars2 = ax.bar(x + width / 2, malicious_vals, width, label=CLASS_NAMES[1], color="coral")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "class_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 7. Summary printout
# ---------------------------------------------------------------------------
def print_summary():
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print("\n" + "=" * 50)
    print("          MODEL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("-" * 50)
    print(report)
    print("=" * 50)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_training_curves()
    plot_epoch_csv_metrics()
    plot_confusion_matrix()
    plot_roc_curve()
    plot_pr_curve()
    plot_class_metrics()
    print_summary()
    print(f"\nAll figures saved in '{OUTPUT_DIR}/' directory.")
