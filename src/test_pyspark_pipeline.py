import os
import sys

# Ensure driver and worker use the Python 3.11 executable
python_path = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

def run_test():
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("FraudDetectionPipelineTest") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    
    csv_path = r"C:\Users\hp\.gemini\antigravity\scratch\credit_card_fraud_detection\data\credit_card_transactions.csv"
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    
    print(f"Loaded records: {df.count()}")
    print("Schema:")
    df.printSchema()

    # 1. Feature Engineering
    # Extract hour from trans_date_trans_time
    df = df.withColumn("trans_hour", F.hour(F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss")))
    df = df.withColumn("trans_dayofweek", F.dayofweek(F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss")))
    
    # Distance between cardholder (lat, long) and merchant (merch_lat, merch_long) using Haversine approximation
    # dlat = merch_lat - lat, dlon = merch_long - long
    df = df.withColumn("distance_km", 
        F.sqrt(
            F.pow(F.col("merch_lat") - F.col("lat"), 2) + 
            F.pow((F.col("merch_long") - F.col("long")) * F.cos(F.toRadians(F.col("lat"))), 2)
        ) * 111.0
    )

    # 2. Outlier removal on amount
    # Let's inspect IQR for amt
    quantiles = df.approxQuantile("amt", [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    upper_bound = q3 + 3.0 * iqr
    print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}, 3x IQR Upper Bound: {upper_bound}")
    
    # Remove extreme non-fraud outliers or apply cap/filter
    # To avoid discarding valuable fraud signal, we remove extreme non-fraud outliers beyond 3x IQR
    # or compare df with/without outlier filtering
    p995 = df.approxQuantile("amt", [0.995], 0.01)[0]
    print(f"99.5th percentile: {p995}")
    
    df_clean = df.filter(~((F.col("amt") > p995) & (F.col("is_fraud") == 0)))
    print(f"Cleaned records: {df_clean.count()}, Fraud count: {df_clean.filter('is_fraud = 1').count()}")

    # 3. Categorical Encoding
    # merchant type / category, gender, state
    cat_indexer = StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep")
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_idx", handleInvalid="keep")
    
    # Label is_fraud as integer/double
    df_clean = df_clean.withColumn("label", F.col("is_fraud").cast(DoubleType()))

    feature_cols = ["amt", "category_idx", "gender_idx", "city_pop", "trans_hour", "trans_dayofweek", "distance_km"]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    
    # Weight column to handle class imbalance in Decision Tree
    fraud_count = df_clean.filter("label = 1").count()
    total_count = df_clean.count()
    fraud_ratio = fraud_count / total_count
    non_fraud_ratio = 1.0 - fraud_ratio
    
    # Weight: fraud weight = 1.0 / fraud_ratio, non-fraud = 1.0 / non_fraud_ratio
    df_clean = df_clean.withColumn("weight", 
        F.when(F.col("label") == 1, F.lit(1.0 / fraud_ratio)).otherwise(F.lit(1.0 / non_fraud_ratio))
    )

    # 4. Train Decision Tree
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="weight",
        maxDepth=5,
        maxBins=32,
        seed=42
    )
    
    pipeline = Pipeline(stages=[cat_indexer, gender_indexer, assembler, dt])
    
    train_data, test_data = df_clean.randomSplit([0.8, 0.2], seed=42)
    print(f"Train size: {train_data.count()}, Test size: {test_data.count()}")
    
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    
    # 5. Evaluate AUC-ROC
    evaluator_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc_roc = evaluator_roc.evaluate(predictions)
    print(f"AUC-ROC Score: {auc_roc:.4f}")
    
    # Multiclass metrics (accuracy, precision, recall, f1)
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    precision_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    
    print(f"Accuracy: {acc_eval.evaluate(predictions):.4f}")
    print(f"F1-Score: {f1_eval.evaluate(predictions):.4f}")
    print(f"Weighted Precision: {precision_eval.evaluate(predictions):.4f}")
    print(f"Weighted Recall: {recall_eval.evaluate(predictions):.4f}")
    
    # Tree model inspect
    dt_model = model.stages[-1]
    print("\nDecision Tree Feature Importances:")
    for col_name, imp in zip(feature_cols, dt_model.featureImportances):
        print(f"  {col_name}: {imp:.4f}")

    # Inspect test fraud confusion
    pred_summary = predictions.groupBy("label", "prediction").count().collect()
    print("\nConfusion Matrix counts:")
    for row in pred_summary:
        print(f"  Label={row['label']}, Pred={row['prediction']} -> Count={row['count']}")

    spark.stop()
    print("Test completed successfully!")

if __name__ == "__main__":
    run_test()
