#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_splits(preprocess_dir: str):
    X_train = pd.read_csv(os.path.join(preprocess_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(preprocess_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(preprocess_dir, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(preprocess_dir, "y_test.csv")).iloc[:, 0]
    preprocess = joblib.load(os.path.join(preprocess_dir, "preprocess.joblib"))

    meta_path = os.path.join(preprocess_dir, "preprocess_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return X_train, X_test, y_train, y_test, preprocess, meta


def main():
    ap = argparse.ArgumentParser(description="Train + MLflow tracking")
    ap.add_argument("--preprocess-dir", type=str, default="artifacts/preprocess", help="Directory from 02_preprocess.py")
    ap.add_argument("--out-dir", type=str, default="artifacts/model", help="Output directory for trained model")
    ap.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI (optional)")
    ap.add_argument("--experiment", type=str, default="weatherAUS", help="MLflow experiment name")
    ap.add_argument("--run-name", type=str, default="logreg_baseline", help="MLflow run name")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)

    mlflow.set_experiment(args.experiment)

    X_train, X_test, y_train, y_test, preprocess, meta = load_splits(args.preprocess_dir)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf),
    ])

    with mlflow.start_run(run_name=args.run_name):
        # Log params
        params = {
            "model": "LogisticRegression",
            "max_iter": 2000,
            "class_weight": "balanced",
            "preprocess_dir": args.preprocess_dir,
        }
        # Merge preprocess meta
        for k, v in (meta or {}).items():
            params[f"data_{k}"] = v

        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Evaluate quickly here (full eval in script 04)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        mlflow.log_metrics(metrics)

        # Save local model
        local_model_path = os.path.join(args.out_dir, "model.joblib")
        joblib.dump(model, local_model_path)

        # Log artifacts
        mlflow.log_artifact(local_model_path, artifact_path="model_artifacts")
        mlflow.sklearn.log_model(model, artifact_path="model_mlflow")

        print("[OK] Training done. Metrics:", metrics)
        print(f"[OK] Model saved to: {local_model_path}")
        print(f"[OK] MLflow run id: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
