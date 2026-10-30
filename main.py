"""
Credit Card Fraud Detection Pipeline Orchestrator (PySpark & MLlib)
===================================================================
Executes the end-to-end Big Data machine learning workflow:
1. SparkSession initialization with Windows socket configuration.
2. Dataset loading and automatic schema / target detection.
3. Missing value imputation and deduplication.
4. Feature engineering: temporal, geospatial Haversine distance, and cardholder age.
5. Stratified train/test split (80/20) preserving class distribution.
6. IQR outlier thresholding fit strictly on training set (preventing data leakage).
7. Balanced class weight calculation for severe class imbalance.
8. ML Pipeline fitting (StringIndexer, OneHotEncoder, VectorAssembler, DecisionTreeClassifier).
9. Test partition evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
10. Extraction of Decision Tree feature importances.
11. Serialization of trained PipelineModel to models/decision_tree_model/.
12. Generation and export of all analytical visualization figures to reports/figures/.
"""

import os
import sys
import time

# Ensure PySpark workers and driver use Python 3.11
PYTHON_311 = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
if os.path.exists(PYTHON_311):
    os.environ["PYSPARK_PYTHON"] = PYTHON_311
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_311

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import functions as F
from src.spark_session import get_spark_session, stop_spark_session
from src.preprocessing import (
    find_dataset_path,
    load_dataset,
    inspect_dataset,
    remove_duplicates_and_impute,
    engineer_features,
    compute_iqr_threshold,
    apply_outlier_treatment
)
from src.model_training import (
    stratified_train_test_split,
    add_class_weights,
    balance_training_data,
    train_decision_tree,
    save_trained_model
)
from src.evaluation import (
    evaluate_model_predictions,
    extract_feature_importances,
    explain_precision_recall_tradeoff,
    save_evaluation_artifacts
)
from src.eda import generate_all_saved_plots


def run_pipeline():
    start_time = time.time()
    print("=" * 75)
    print(" CREDIT CARD FRAUD DETECTION END-TO-END PIPELINE")
    print(" Technology: Apache PySpark MLlib | Model: Decision Tree Classifier")
    print("=" * 75)

    # 1. Initialize SparkSession
    print("\n[STEP 1/7] Initializing PySpark Session on Windows...")
    spark = get_spark_session("CreditCardFraudDetectionPipeline", master="local[2]")
    print(f"Spark Version: {spark.version} | Driver Python: {os.environ.get('PYSPARK_PYTHON', sys.executable)}")

    try:
        # 2. Locate and Load Dataset
        print("\n[STEP 2/7] Locating and loading transaction dataset...")
        csv_path = find_dataset_path()
        print(f"Dataset path: {csv_path}")
        raw_df = load_dataset(spark, csv_path)
        inspect_dataset(raw_df, num_records=5)

        # 3. Clean and Engineer Features
        print("\n[STEP 3/7] Cleaning data and engineering behavioral features...")
        clean_df = remove_duplicates_and_impute(raw_df)
        feat_df, feature_meta = engineer_features(clean_df)
        feat_df.cache()

        print(f"Engineered numerical features: {feature_meta['numeric_cols']}")
        print(f"Engineered categorical features: {feature_meta['categorical_cols']}")

        # 4. Stratified Split and Leakage-Free Outlier Capping
        print("\n[STEP 4/7] Splitting data into stratified 80% train / 20% test partitions...")
        train_df, test_df = stratified_train_test_split(feat_df, train_ratio=0.8, seed=42)

        # Fit IQR on training split only
        iqr_stats = compute_iqr_threshold(train_df, column="amt", factor=3.0)
        print(f"IQR Amount Capping threshold (fit on train data): ${iqr_stats['upper_threshold']:.2f}")

        # Apply outlier treatment (preserving actual amounts for fraudulent transactions)
        train_df = apply_outlier_treatment(train_df, "amt", iqr_stats, preserve_fraud=True)
        test_df = apply_outlier_treatment(test_df, "amt", iqr_stats, preserve_fraud=True)

        # Class balancing & weighting
        train_df_balanced = balance_training_data(train_df, oversample_ratio=0.15, seed=42)
        train_df_weighted, weight_info = add_class_weights(train_df_balanced)
        test_df = test_df.withColumn("weight", F.lit(1.0))

        print(f"Computed Class Weights: Fraud = {weight_info['weight_fraud']:.2f}, Legit = {weight_info['weight_legit']:.4f}")

        # 5. Train Decision Tree Classifier Pipeline
        print("\n[STEP 5/7] Training PySpark ML Pipeline with Decision Tree Classifier...")
        pipeline_model, pipeline_def, training_meta = train_decision_tree(
            train_df=train_df_weighted,
            feature_meta=feature_meta,
            max_depth=5,
            max_bins=32,
            min_instances_per_node=2,
            impurity="gini",
            use_class_weights=True
        )

        # 6. Evaluate Model on Test Partition
        print("\n[STEP 6/7] Evaluating model performance on unseen test partition...")
        _, eval_results = evaluate_model_predictions(pipeline_model, test_df)
        metrics = eval_results["metrics"]
        roc_data = eval_results["roc_data"]

        # Extract Feature Importances
        fi_df = extract_feature_importances(pipeline_model, feature_meta)

        # 7. Persist Model and Generate Artifacts
        print("\n[STEP 7/7] Persisting trained model and generating analytical artifacts...")
        models_dir = os.path.join(PROJECT_ROOT, "models")
        model_save_path = os.path.join(models_dir, "decision_tree_model")
        save_trained_model(pipeline_model, model_save_path)

        # Save metrics, roc data, feature importance
        save_evaluation_artifacts(metrics, roc_data, fi_df, models_dir)

        # Generate saved figures for reports
        figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
        # Convert raw dataset to Pandas for analytical plotting
        raw_pandas_df = raw_df.toPandas()
        if "trans_date_trans_time" in raw_pandas_df.columns:
            raw_pandas_df["trans_hour"] = pd.to_datetime(raw_pandas_df["trans_date_trans_time"]).dt.hour
        if "amt" in raw_pandas_df.columns and "is_fraud" in raw_pandas_df.columns:
            generate_all_saved_plots(raw_pandas_df, metrics, roc_data, fi_df, figures_dir)

        elapsed = time.time() - start_time
        print("\n" + "=" * 75)
        print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
        print(f" Saved Model:       {model_save_path}")
        print(f" Evaluation Report: {os.path.join(models_dir, 'metrics.json')}")
        print(f" Visualizations:    {figures_dir}")
        print(f" Accuracy:          {metrics['accuracy'] * 100:.2f}%")
        print(f" ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f" Fraud Recall:      {metrics['recall'] * 100:.2f}%")
        print(f" Fraud Precision:   {metrics['precision'] * 100:.2f}%")
        print("=" * 75)

    finally:
        print("\nStopping PySpark Session...")
        stop_spark_session()
        print("PySpark Session terminated cleanly.")


if __name__ == "__main__":
    import pandas as pd
    run_pipeline()
