"""
Model Training Module
=====================
Implements stratified dataset splitting, class weight computation,
PySpark ML Pipeline assembly, Decision Tree Classifier training,
hyperparameter configuration, and model persistence to models/decision_tree_model/.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import DecisionTreeClassifier

from src.preprocessing import (
    build_pipeline_stages,
    compute_iqr_threshold,
    apply_outlier_treatment
)


def stratified_train_test_split(
    df: DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[DataFrame, DataFrame]:
    """
    Performs stratified split preserving the exact ratio of legitimate
    and fraudulent transactions in both training and test partitions.
    """
    fraud_df = df.filter(F.col("label") == 1.0)
    legit_df = df.filter(F.col("label") == 0.0)

    test_ratio = 1.0 - train_ratio
    train_fraud, test_fraud = fraud_df.randomSplit([train_ratio, test_ratio], seed=seed)
    train_legit, test_legit = legit_df.randomSplit([train_ratio, test_ratio], seed=seed)

    train_df = train_fraud.unionByName(train_legit)
    test_df = test_fraud.unionByName(test_legit)

    return train_df, test_df


def add_class_weights(train_df: DataFrame) -> Tuple[DataFrame, Dict[str, float]]:
    """
    Computes balanced class weights for imbalanced fraud data:
    weight_class = total_records / (2 * class_records)
    Adds a 'weight' column to the training DataFrame for DecisionTreeClassifier.
    """
    total_count = train_df.count()
    fraud_count = train_df.filter(F.col("label") == 1.0).count()
    legit_count = train_df.filter(F.col("label") == 0.0).count()

    w_fraud = float(total_count) / (2.0 * max(fraud_count, 1))
    w_legit = float(total_count) / (2.0 * max(legit_count, 1))

    weighted_train_df = train_df.withColumn(
        "weight",
        F.when(F.col("label") == 1.0, F.lit(w_fraud)).otherwise(F.lit(w_legit))
    )

    weights_info = {
        "weight_fraud": w_fraud,
        "weight_legit": w_legit,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "total_count": total_count
    }

    return weighted_train_df, weights_info


def balance_training_data(
    train_df: DataFrame,
    oversample_ratio: float = 0.15,
    seed: int = 42
) -> DataFrame:
    """
    Optional stratified oversampling for severe class imbalance (e.g. <0.5% fraud).
    Synthetically replicates minority fraud cases to ensure decision tree splits
    adequately capture fraud rules while preserving original test data untouched.
    """
    fraud_df = train_df.filter(F.col("label") == 1.0)
    legit_df = train_df.filter(F.col("label") == 0.0)

    n_fraud = fraud_df.count()
    n_legit = legit_df.count()

    target_fraud_count = int(n_legit * oversample_ratio)
    if target_fraud_count > n_fraud and n_fraud > 0:
        multiplier = target_fraud_count // n_fraud
        oversampled_fraud = fraud_df
        for _ in range(max(1, multiplier - 1)):
            oversampled_fraud = oversampled_fraud.unionByName(fraud_df)
        train_df = oversampled_fraud.unionByName(legit_df)

    return train_df


def train_decision_tree(
    train_df: DataFrame,
    feature_meta: Dict[str, Any],
    max_depth: int = 5,
    max_bins: int = 32,
    min_instances_per_node: int = 2,
    impurity: str = "gini",
    use_class_weights: bool = True
) -> Tuple[PipelineModel, Pipeline, Dict[str, Any]]:
    """
    Constructs and fits a PySpark ML Pipeline with feature encoders, assembler,
    and a DecisionTreeClassifier.
    """
    numeric_cols = feature_meta["numeric_cols"]
    categorical_cols = feature_meta["categorical_cols"]

    # 1. Build preprocessing stages
    stages, assembled_inputs = build_pipeline_stages(categorical_cols, numeric_cols)

    # 2. Add DecisionTreeClassifier stage
    dt_params = {
        "featuresCol": "features",
        "labelCol": "label",
        "predictionCol": "prediction",
        "probabilityCol": "probability",
        "rawPredictionCol": "rawPrediction",
        "maxDepth": max_depth,
        "maxBins": max_bins,
        "minInstancesPerNode": min_instances_per_node,
        "impurity": impurity,
        "seed": 42
    }

    if use_class_weights and "weight" in train_df.columns:
        dt_params["weightCol"] = "weight"

    dt = DecisionTreeClassifier(**dt_params)
    stages.append(dt)

    pipeline = Pipeline(stages=stages)

    print(f"\n[MODEL TRAINING] Fitting PySpark DecisionTree Pipeline...")
    print(f"Hyperparameters: maxDepth={max_depth}, maxBins={max_bins}, "
          f"minInstancesPerNode={min_instances_per_node}, impurity={impurity}")

    pipeline_model = pipeline.fit(train_df)
    print("[MODEL TRAINING] Model successfully trained.")

    training_meta = {
        "max_depth": max_depth,
        "max_bins": max_bins,
        "min_instances_per_node": min_instances_per_node,
        "impurity": impurity,
        "assembled_inputs": assembled_inputs,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }

    return pipeline_model, pipeline, training_meta


def save_trained_model(pipeline_model: PipelineModel, save_path: str):
    """
    Persists the fitted PySpark PipelineModel to disk inside models/decision_tree_model/.
    Ensures the model can be loaded again without retraining.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    pipeline_model.write().overwrite().save(save_path)
    print(f"[MODEL PERSISTENCE] PipelineModel saved to: {os.path.abspath(save_path)}")


def load_saved_model(load_path: str) -> PipelineModel:
    """
    Loads a previously saved PySpark PipelineModel from disk.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model path does not exist: {load_path}")
    return PipelineModel.load(load_path)


if __name__ == "__main__":
    from src.spark_session import get_spark_session, stop_spark_session
    from src.preprocessing import find_dataset_path, load_dataset, remove_duplicates_and_impute, engineer_features

    spark = get_spark_session("ModelTrainingTest")
    try:
        csv_path = find_dataset_path()
        df = load_dataset(spark, csv_path)
        df_clean = remove_duplicates_and_impute(df)
        df_feat, meta = engineer_features(df_clean)

        train_df, test_df = stratified_train_test_split(df_feat, train_ratio=0.8)
        iqr_stats = compute_iqr_threshold(train_df, column="amt", factor=3.0)
        train_df = apply_outlier_treatment(train_df, "amt", iqr_stats, preserve_fraud=True)
        test_df = apply_outlier_treatment(test_df, "amt", iqr_stats, preserve_fraud=True)

        train_df_balanced = balance_training_data(train_df, oversample_ratio=0.15)
        train_df_weighted, _ = add_class_weights(train_df_balanced)
        test_df = test_df.withColumn("weight", F.lit(1.0))

        model, _, _ = train_decision_tree(train_df_weighted, meta, max_depth=5)
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "decision_tree_model"))
        save_trained_model(model, save_path)
        print("[SUCCESS] Model trained and saved.")
    finally:
        stop_spark_session()
