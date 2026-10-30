import os
import sys

python_path = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .master("local[1]") \
    .appName("TuningFraudPipeline") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

csv_path = r"C:\Users\hp\.gemini\antigravity\scratch\credit_card_fraud_detection\data\credit_card_transactions.csv"
df = spark.read.csv(csv_path, header=True, inferSchema=True)

# Add spending behavior features
df = df.withColumn("trans_time_ts", F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"))
df = df.withColumn("trans_hour", F.hour("trans_time_ts"))
df = df.withColumn("trans_dayofweek", F.dayofweek("trans_time_ts"))
df = df.withColumn("is_night", F.when((F.col("trans_hour") >= 22) | (F.col("trans_hour") <= 5), 1.0).otherwise(0.0))

# Distance km
df = df.withColumn("distance_km", 
    F.sqrt(
        F.pow(F.col("merch_lat") - F.col("lat"), 2) + 
        F.pow((F.col("merch_long") - F.col("long")) * F.cos(F.radians(F.col("lat"))), 2)
    ) * 111.0
)

# Cardholder age
df = df.withColumn("dob_date", F.to_date("dob", "yyyy-MM-dd"))
df = df.withColumn("age", (F.datediff(F.col("trans_time_ts"), F.col("dob_date")) / 365.25).cast(DoubleType()))

# Outlier handling:
# Let's inspect capping or filtering extreme non-fraud recording anomalies (amt > 3000)
df_clean = df.filter(~((F.col("amt") > 3000) & (F.col("is_fraud") == 0)))

# Index categoricals
cat_indexer = StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep")
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_idx", handleInvalid="keep")

df_clean = df_clean.withColumn("label", F.col("is_fraud").cast(DoubleType()))

feature_cols = ["amt", "category_idx", "gender_idx", "city_pop", "trans_hour", "is_night", "trans_dayofweek", "distance_km", "age"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

# Stratified split: 75% train, 25% test
frauds = df_clean.filter("label = 1")
non_frauds = df_clean.filter("label = 0")

train_f, test_f = frauds.randomSplit([0.75, 0.25], seed=42)
train_nf, test_nf = non_frauds.randomSplit([0.75, 0.25], seed=42)

# Oversample fraud in train set
multiplier = int(train_nf.count() / (train_f.count() * 10))  # e.g. bring fraud to ~10% of train set
print(f"Oversample multiplier: {multiplier}")
train_f_oversampled = train_f
for _ in range(multiplier):
    train_f_oversampled = train_f_oversampled.union(train_f)

train_data = train_nf.union(train_f_oversampled)
test_data = test_nf.union(test_f)

print(f"Train size: {train_data.count()} (frauds: {train_data.filter('label=1').count()})")
print(f"Test size: {test_data.count()} (frauds: {test_data.filter('label=1').count()})")

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    minInstancesPerNode=2,
    seed=42
)

pipeline = Pipeline(stages=[cat_indexer, gender_indexer, assembler, dt])
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

evaluator_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_roc = evaluator_roc.evaluate(predictions)
print(f"AUC-ROC Score with rawPrediction: {auc_roc:.4f}")

evaluator_roc_prob = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC")
print(f"AUC-ROC Score with probability: {evaluator_roc_prob.evaluate(predictions):.4f}")

acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
print(f"Accuracy: {acc_eval.evaluate(predictions):.4f}")
print(f"F1-Score: {f1_eval.evaluate(predictions):.4f}")

# Confusion matrix
cm = predictions.groupBy("label", "prediction").count().collect()
print("Confusion Matrix:")
for r in cm:
    print(f"  True={r['label']}, Pred={r['prediction']} -> Count={r['count']}")

spark.stop()
