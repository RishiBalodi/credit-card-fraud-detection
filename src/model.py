import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Default feature definitions
CATEGORICAL_COLS = ["category", "gender"]
NUMERIC_COLS = ["amt", "city_pop", "trans_hour", "is_night", "trans_dayofweek", "distance_km", "age"]

def build_pipeline(feature_cols=None, max_depth=5, min_instances_per_node=2, max_bins=32):
    """
    Builds a PySpark ML Pipeline with StringIndexers, VectorAssembler,
    and a DecisionTreeClassifier.
    """
    if feature_cols is None:
        feature_cols = NUMERIC_COLS

    stages = []
    
    # 1. StringIndexers for categorical features
    cat_indexed_cols = []
    for cat_col in CATEGORICAL_COLS:
        idx_col = f"{cat_col}_idx"
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=idx_col,
            handleInvalid="keep"
        )
        stages.append(indexer)
        cat_indexed_cols.append(idx_col)

    # 2. VectorAssembler combining numeric + categorical index columns
    all_features = feature_cols + cat_indexed_cols
    assembler = VectorAssembler(
        inputCols=all_features,
        outputCol="features",
        handleInvalid="skip"
    )
    stages.append(assembler)

    # 3. Decision Tree Classifier
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        maxDepth=max_depth,
        minInstancesPerNode=min_instances_per_node,
        maxBins=max_bins,
        seed=42
    )
    stages.append(dt)

    pipeline = Pipeline(stages=stages)
    return pipeline, all_features

def train_and_evaluate(df, train_ratio=0.8, oversample_fraud=True, max_depth=5, min_instances=2):
    """
    Trains a PySpark Decision Tree model on stratified training data and evaluates on test data.
    
    Returns:
    - model: Fitted PipelineModel
    - metrics: Dictionary of evaluation metrics
    - roc_data: Dictionary of FPR, TPR, thresholds for plotting
    - predictions_df: Pandas DataFrame of test set predictions for interactive inspection
    - feature_importance_df: Pandas DataFrame of feature names and importances
    - tree_debug: Text string of the trained Decision Tree logic
    """
    # 1. Stratified train/test split on label
    frauds = df.filter(F.col("label") == 1.0)
    non_frauds = df.filter(F.col("label") == 0.0)

    test_ratio = 1.0 - train_ratio
    train_f, test_f = frauds.randomSplit([train_ratio, test_ratio], seed=42)
    train_nf, test_nf = non_frauds.randomSplit([train_ratio, test_ratio], seed=42)

    # Minority oversampling on training data if requested
    if oversample_fraud:
        f_count = train_f.count()
        nf_count = train_nf.count()
        if f_count > 0:
            target_multiplier = max(1, min(30, int(nf_count / (f_count * 10))))
            train_f_rep = train_f.withColumn(
                "_mult", F.explode(F.array([F.lit(i) for i in range(target_multiplier)]))
            ).drop("_mult")
            train_data = train_nf.union(train_f_rep)
        else:
            train_data = train_nf.union(train_f)
    else:
        train_data = train_nf.union(train_f)

    test_data = test_nf.union(test_f)

    # 2. Build and fit pipeline
    pipeline, feature_names = build_pipeline(
        max_depth=max_depth,
        min_instances_per_node=min_instances
    )
    model = pipeline.fit(train_data)

    # 3. Predict on unseen test data
    predictions = model.transform(test_data)

    # 4. Evaluate AUC-ROC in PySpark
    eval_roc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="probability",
        metricName="areaUnderROC"
    )
    auc_roc = float(eval_roc.evaluate(predictions))

    eval_pr = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="probability",
        metricName="areaUnderPR"
    )
    auc_pr = float(eval_pr.evaluate(predictions))

    # Multiclass evaluators
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    accuracy = float(acc_eval.evaluate(predictions))
    f1 = float(f1_eval.evaluate(predictions))
    precision = float(prec_eval.evaluate(predictions))
    recall = float(rec_eval.evaluate(predictions))

    # 5. Extract test predictions to pandas for ROC curve coordinates and detailed metrics
    pred_sample = predictions.select(
        "label",
        "prediction",
        "probability",
        "amt",
        "category",
        "trans_hour",
        "distance_km"
    ).toPandas()

    # Extract probability of class 1 (Fraud) from Spark DenseVector
    pred_sample["prob_1"] = pred_sample["probability"].apply(lambda v: float(v[1]) if hasattr(v, "__getitem__") else float(v))

    y_true = pred_sample["label"].values.astype(int)
    y_prob = pred_sample["prob_1"].values
    y_pred = pred_sample["prediction"].values.astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    sk_auc = float(auc(fpr, tpr))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])) if cm.shape == (2, 2) else (int(cm[0, 0]), 0, 0, 0)
    
    # Specificity and Sensitivity
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fraud_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "auc_roc_sklearn": round(sk_auc, 4),
        "auc_pr": round(auc_pr, 4),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fraud_precision": round(fraud_precision, 4),
        "fraud_recall": round(sensitivity, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "train_records": train_data.count(),
        "test_records": test_data.count(),
        "train_frauds": train_data.filter("label = 1").count(),
        "test_frauds": int(np.sum(y_true == 1))
    }

    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
        "auc": round(sk_auc, 4)
    }

    # Extract Feature Importances from the DecisionTree stage
    dt_stage = model.stages[-1]
    importances = dt_stage.featureImportances.toArray()
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    fi_df["importance_pct"] = (fi_df["importance"] * 100).round(2)

    tree_debug = dt_stage.toDebugString

    return model, metrics, roc_data, pred_sample, fi_df, tree_debug
