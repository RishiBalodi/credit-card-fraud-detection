"""
Credit Card Fraud Detection Pipeline using PySpark
=================================================
Context: Identify fraudulent transactions based on spending behavior.

Tasks completed:
1. Load transaction data and remove outliers.
2. Convert categorical features like merchant type to numeric values (StringIndexer).
3. Train a Decision Tree classifier to classify fraud vs non-fraud.
4. Evaluate model performance with AUC-ROC curve and classification metrics.
5. Visualize fraud distribution by category and spending behavior.
"""

import os
import sys
import json
import time

# Ensure Spark workers and driver use the correct Python interpreter
PYTHON_EXEC = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
if os.path.exists(PYTHON_EXEC):
    os.environ["PYSPARK_PYTHON"] = PYTHON_EXEC
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXEC

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.spark_session import get_spark_session, stop_spark_session
from src.data_preprocessing import load_data, engineer_features, detect_and_remove_outliers
from src.model import train_and_evaluate
from src.visualization import (
    plot_fraud_distribution_by_category,
    plot_auc_roc_curve,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_spending_behavior
)

def main():
    print("=" * 70)
    print(" CREDIT CARD FRAUD DETECTION PIPELINE (PySpark)")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Initialize PySpark Session
    print("\n[1/6] Initializing SparkSession...")
    spark = get_spark_session("CreditCardFraudDetection", master="local[2]")
    print(f"Spark Version: {spark.version}")

    # Step 2: Load Transaction Data
    csv_path = os.path.join(BASE_DIR, "data", "credit_card_transactions.csv")
    print(f"\n[2/6] Loading transaction dataset from: {csv_path}")
    raw_df = load_data(spark, csv_path)
    total_raw = raw_df.count()
    print(f"Dataset successfully loaded. Total transactions: {total_raw:,}")

    # Step 3: Feature Engineering
    print("\n[3/6] Engineering spending behavior and transaction features...")
    df_feat = engineer_features(raw_df)
    
    # Cache intermediate dataframe
    df_feat.cache()

    # Step 4: Outlier Detection and Removal
    print("\n[4/6] Detecting and removing spending outliers (IQR method on transaction amount)...")
    cleaned_df, outlier_stats = detect_and_remove_outliers(
        df_feat, column="amt", method="iqr", factor=3.0, preserve_fraud=True
    )
    cleaned_df.cache()

    print(f"  • Outlier Method: {outlier_stats['method'].upper()} (Factor: {outlier_stats['factor']}x)")
    print(f"  • Outlier Thresholds: Lower Bound = ${outlier_stats['lower_bound']:.2f}, Upper Bound = ${outlier_stats['upper_bound']:.2f}")
    print(f"  • Outliers Removed: {outlier_stats['outliers_removed']} rows ({outlier_stats['percentage_removed']}%)")
    print(f"  • Remaining Transactions: {outlier_stats['total_after']:,}")
    print(f"  • Fraud Records Preserved: {outlier_stats['fraud_after']}/{outlier_stats['fraud_before']} ({outlier_stats['fraud_preserved_pct']}%)")

    # Step 5: Convert Categorical Features & Train Decision Tree Classifier
    print("\n[5/6] Building PySpark ML Pipeline & Training Decision Tree Classifier...")
    print("  • Indexing categorical feature 'category' (Merchant Type) and 'gender' via StringIndexer")
    print("  • Assembling numeric and indexed behavioral features with VectorAssembler")
    print("  • Training DecisionTreeClassifier (maxDepth=5, minInstancesPerNode=2)...")

    model, metrics, roc_data, pred_sample, fi_df, tree_debug = train_and_evaluate(
        cleaned_df,
        train_ratio=0.8,
        oversample_fraud=True,
        max_depth=5,
        min_instances=2
    )

    # Step 6: Model Evaluation & Performance Metrics
    print("\n[6/6] Model Evaluation & AUC-ROC Curve Calculation:")
    print("-" * 50)
    print(f"  • Area Under ROC (AUC-ROC):  {metrics['auc_roc']:.4f} (PySpark)")
    print(f"  • Area Under PR (AUC-PR):    {metrics['auc_pr']:.4f}")
    print(f"  • Overall Accuracy:          {metrics['accuracy'] * 100:.2f}%")
    print(f"  • Weighted F1-Score:         {metrics['f1_score']:.4f}")
    print(f"  • Weighted Precision:        {metrics['precision']:.4f}")
    print(f"  • Weighted Recall:           {metrics['recall']:.4f}")
    print(f"  • Fraud Sensitivity/Recall:  {metrics['fraud_recall'] * 100:.2f}%")
    print(f"  • Fraud Precision:           {metrics['fraud_precision'] * 100:.2f}%")
    print(f"  • Non-Fraud Specificity:     {metrics['specificity'] * 100:.2f}%")
    cm = metrics["confusion_matrix"]
    print(f"  • Confusion Matrix:")
    print(f"      True Negatives (Legit -> Legit): {cm['tn']:,}")
    print(f"      False Positives (Legit -> Fraud): {cm['fp']}")
    print(f"      False Negatives (Fraud -> Legit): {cm['fn']}")
    print(f"      True Positives (Fraud -> Fraud):  {cm['tp']}")
    print("-" * 50)

    # Step 7: Visualizations & Saving Artifacts
    print("\n[Visualizations] Generating and saving analytical plots...")
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Convert a sample of cleaned_df to Pandas for plotting
    df_pandas = cleaned_df.select("amt", "category", "is_fraud", "trans_hour", "distance_km", "gender").toPandas()

    # 1. Fraud distribution by category
    cat_fig_path = os.path.join(models_dir, "fraud_distribution_by_category.png")
    fig_cat, cat_summary = plot_fraud_distribution_by_category(df_pandas, save_path=cat_fig_path)
    print(f"  Saved: {cat_fig_path}")

    # 2. AUC-ROC Curve
    roc_fig_path = os.path.join(models_dir, "auc_roc_curve.png")
    fig_roc = plot_auc_roc_curve(roc_data, save_path=roc_fig_path)
    print(f"  Saved: {roc_fig_path}")

    # 3. Confusion Matrix
    cm_fig_path = os.path.join(models_dir, "confusion_matrix.png")
    fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_fig_path)
    print(f"  Saved: {cm_fig_path}")

    # 4. Feature Importance
    fi_fig_path = os.path.join(models_dir, "feature_importance.png")
    fig_fi = plot_feature_importances(fi_df, save_path=fi_fig_path)
    print(f"  Saved: {fi_fig_path}")

    # 5. Spending behavior
    sb_fig_path = os.path.join(models_dir, "spending_behavior.png")
    fig_sb = plot_spending_behavior(df_pandas, save_path=sb_fig_path)
    print(f"  Saved: {sb_fig_path}")

    # Save metrics and artifacts to JSON
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    roc_path = os.path.join(models_dir, "roc_data.json")
    with open(roc_path, "w") as f:
        json.dump(roc_data, f, indent=2)

    fi_path = os.path.join(models_dir, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)

    cat_path = os.path.join(models_dir, "category_summary.csv")
    cat_summary.to_csv(cat_path, index=False)

    print(f"  Saved evaluation metrics to: {metrics_path}")

    # Display Top Category Risk Summary
    print("\n[Summary] Top Merchant Categories by Fraud Incident Rate:")
    print(cat_summary[["category", "total_trans", "fraud_trans", "fraud_rate", "avg_amt"]].to_string(index=False))

    # Top Feature Importances
    print("\n[Summary] Decision Tree Feature Importances:")
    print(fi_df[["feature", "importance_pct"]].to_string(index=False))

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 70)

    stop_spark_session()

if __name__ == "__main__":
    main()
