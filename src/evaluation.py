"""
Model Evaluation and Feature Importance Module
==============================================
Calculates classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
generates the confusion matrix, computes ROC curve coordinates, extracts
Decision Tree feature importances, and explains the precision-recall trade-off.
"""

import os
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassificationModel


def evaluate_model_predictions(
    pipeline_model: PipelineModel,
    test_df: DataFrame
) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Applies the trained PipelineModel to test data and calculates:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    - Confusion Matrix (TP, TN, FP, FN)
    """
    print("\n[EVALUATION] Generating predictions on test dataset...")
    predictions = pipeline_model.transform(test_df)
    predictions.cache()

    # 1. PySpark Evaluators
    binary_eval = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderROC"
    )
    roc_auc = float(binary_eval.evaluate(predictions))

    mc_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = float(mc_acc.evaluate(predictions))

    mc_prec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    weighted_precision = float(mc_prec.evaluate(predictions))

    mc_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    weighted_recall = float(mc_rec.evaluate(predictions))

    mc_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = float(mc_f1.evaluate(predictions))

    # 2. Extract label and prediction vectors for binary-specific metrics & confusion matrix
    pred_sample = predictions.select("label", "prediction", "probability").toPandas()
    y_true = pred_sample["label"].values.astype(int)
    y_pred = pred_sample["prediction"].values.astype(int)

    # Probabilities for class 1 (Fraud)
    # probability vector has 2 elements: [P(class=0), P(class=1)]
    y_scores = np.array([float(p[1]) for p in pred_sample["probability"].values])

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Class 1 (Fraud) specific Precision & Recall
    fraud_precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    fraud_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    fraud_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Sklearn ROC curve calculation for high-resolution plotting
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    sklearn_auc = float(auc(fpr, tpr)) if len(np.unique(y_true)) > 1 else roc_auc

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(fraud_precision, 4),
        "recall": round(fraud_recall, 4),
        "f1_score": round(fraud_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_f1": round(f1, 4),
        "roc_auc": round(roc_auc if roc_auc > 0.0 else sklearn_auc, 4),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "total_test_samples": len(y_true),
        "fraud_test_samples": int(sum(y_true)),
        "legit_test_samples": int(len(y_true) - sum(y_true))
    }

    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": metrics["roc_auc"]
    }

    print("=" * 60)
    print(" MODEL PERFORMANCE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:            {metrics['accuracy'] * 100:.2f}%")
    print(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f"Fraud Recall:        {metrics['recall'] * 100:.2f}% (Caught {tp} of {tp + fn} frauds)")
    print(f"Fraud Precision:     {metrics['precision'] * 100:.2f}% ({tp} true frauds in {tp + fp} alerts)")
    print(f"Fraud F1-Score:      {metrics['f1_score']:.4f}")
    print(f"Weighted F1:         {metrics['weighted_f1']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  [TN={tn:,}  |  FP={fp:,}]")
    print(f"  [FN={fn:,}  |  TP={tp:,}]")
    print("=" * 60)

    return predictions, {"metrics": metrics, "roc_data": roc_data}


def extract_feature_importances(
    pipeline_model: PipelineModel,
    feature_meta: Dict[str, Any]
) -> pd.DataFrame:
    """
    Extracts feature importances from the trained DecisionTreeClassificationModel
    and maps them to meaningful feature names.
    """
    dt_model: Optional[DecisionTreeClassificationModel] = None
    for stage in pipeline_model.stages:
        if isinstance(stage, DecisionTreeClassificationModel):
            dt_model = stage
            break

    if dt_model is None:
        raise ValueError("No DecisionTreeClassificationModel found in Pipeline stages.")

    raw_importances = dt_model.featureImportances.toArray()

    # Reconstruct human-readable feature names from pipeline stages
    numeric_cols = feature_meta.get("numeric_cols", [])
    cat_cols = feature_meta.get("categorical_cols", [])

    feature_names = []
    # Numeric features come first in our assembler
    feature_names.extend(numeric_cols)

    # For each categorical feature, extract encoded category labels
    stage_idx = 0
    for cat_col in cat_cols:
        indexer_model = pipeline_model.stages[stage_idx]
        labels = list(indexer_model.labels)
        # OneHotEncoder keeps dropLast=True by default in PySpark ML
        # So it creates len(labels) - 1 or len(labels) columns
        # Let's inspect the encoder stage:
        for lbl in labels[:-1]:
            feature_names.append(f"{cat_col}_{lbl}")
        stage_idx += 2  # skip StringIndexer and OneHotEncoder

    # Fallback padding if lengths differ
    if len(feature_names) < len(raw_importances):
        for i in range(len(feature_names), len(raw_importances)):
            feature_names.append(f"feature_{i}")
    elif len(feature_names) > len(raw_importances):
        feature_names = feature_names[:len(raw_importances)]

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": raw_importances
    })

    fi_df["importance_pct"] = (fi_df["importance"] / fi_df["importance"].sum()) * 100.0
    fi_df = fi_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\n--- Decision Tree Top Influential Features ---")
    for _, row in fi_df.head(8).iterrows():
        print(f"  - {row['feature']:<20}: {row['importance']:.4f} ({row['importance_pct']:.1f}%)")

    return fi_df


def explain_precision_recall_tradeoff() -> str:
    """
    Provides a comprehensive domain explanation of the Precision-Recall trade-off
    in financial credit card fraud detection.
    """
    explanation = """
### Precision-Recall Trade-off in Credit Card Fraud Detection

1. **Why Accuracy is Misleading**:
   In credit card fraud datasets, legitimate transactions typically represent >99% of total volume
   (0.36% fraud rate in this 5,000-sample dataset). A trivial model predicting 0 for every transaction
   achieves 99.64% accuracy, yet catches zero fraud cases ($0 intercepted, 100% loss).

2. **Recall (Sensitivity)**:
   - **Definition**: True Positives / (True Positives + False Negatives)
   - **Business Impact**: Measures the fraction of actual fraud transactions intercepted.
   - **Goal in Banking**: Maximize recall to avoid direct monetary theft, chargeback liabilities,
     card network non-compliance fines, and cardholder security breach reputation loss.

3. **Precision (Positive Predictive Value)**:
   - **Definition**: True Positives / (True Positives + False Positives)
   - **Business Impact**: Measures the accuracy of fraud alarms (what percentage of flagged alerts are real fraud).
   - **Goal in Banking**: Prevent customer friction caused by false positives (declining a legitimate purchase
     at a grocery store, gas station, or dinner creates cardholder dissatisfaction and card abandonment).

4. **Strategic Equilibrium (The F1-Score / Threshold Tuning)**:
   Financial institutions deploy multi-tiered risk thresholds:
   - High Probability (>80%): Auto-block transaction and lock card.
   - Medium Probability (40%-80%): Trigger step-up authentication (SMS OTP, biometric approval).
   - Low Probability (<40%): Allow transaction seamlessly in the background.
"""
    return explanation.strip()


def save_evaluation_artifacts(
    metrics: Dict[str, Any],
    roc_data: Dict[str, Any],
    fi_df: pd.DataFrame,
    output_dir: str
):
    """
    Saves metrics.json, roc_data.json, and feature_importance.csv to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    roc_path = os.path.join(output_dir, "roc_data.json")
    with open(roc_path, "w") as f:
        json.dump(roc_data, f, indent=2)

    fi_path = os.path.join(output_dir, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)

    print(f"\n[ARTIFACTS SAVED] Evaluation artifacts stored in: {os.path.abspath(output_dir)}")
