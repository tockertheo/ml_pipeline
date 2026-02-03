#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained model")
    ap.add_argument("--preprocess-dir", type=str, default="artifacts/preprocess", help="Directory from 02_preprocess.py")
    ap.add_argument("--model-path", type=str, default="artifacts/model/model.joblib", help="Path to model.joblib")
    ap.add_argument("--out-dir", type=str, default="artifacts/eval", help="Output directory for evaluation artifacts")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Load test split
    X_test = pd.read_csv(os.path.join(args.preprocess_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(args.preprocess_dir, "y_test.csv")).iloc[:, 0]

    # Load model
    model = joblib.load(args.model_path)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    # Classification report
    report = classification_report(y_test, y_pred, digits=4)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(args.out_dir, "confusion_matrix.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, indent=2)

    # Plot confusion matrix (simple)
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # Save metrics
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] Evaluation artifacts written to:", args.out_dir)
    print("[OK] Metrics:", metrics)


if __name__ == "__main__":
    main()
