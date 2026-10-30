"""
Data Preprocessing and Feature Engineering Module
=================================================
Handles data loading via PySpark, schema inspection, missing value handling,
duplicate removal, PII filtering, temporal and geospatial feature engineering,
IQR-based outlier capping (fitted on training data to prevent leakage),
and PySpark ML feature pipeline assembly.
"""

import os
import sys
from typing import Tuple, List, Dict, Any, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, TimestampType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler


def find_dataset_path(preferred_name: str = "fraudTest_5000.csv", search_dir: Optional[str] = None) -> str:
    """
    Locates the dataset automatically, searching for fraudTest_5000.csv,
    credit_card_transactions.csv, or any suitable CSV in the data directory.
    """
    if search_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        search_dir = os.path.join(base_dir, "data")

    candidates = [
        os.path.join(search_dir, preferred_name),
        os.path.join(search_dir, "credit_card_transactions.csv"),
        os.path.join(search_dir, "fraudTest.csv"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)

    # Search directory for any CSV with 'fraud' or 'trans' in name
    if os.path.exists(search_dir):
        for f in os.listdir(search_dir):
            if f.endswith(".csv"):
                return os.path.abspath(os.path.join(search_dir, f))

    raise FileNotFoundError(
        f"Could not find any suitable credit card transaction dataset in '{search_dir}'. "
        "Please place 'fraudTest_5000.csv' in the 'data/' directory."
    )


def detect_target_column(df: DataFrame) -> str:
    """
    Detects the target/label column name dynamically from dataset schema.
    Supports 'is_fraud', 'Class', 'fraud', 'isFraud', 'label'.
    """
    candidates = ["is_fraud", "Class", "class", "fraud", "isFraud", "label", "target"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No valid fraud target column found. Available columns: {df.columns}")


def load_dataset(spark: SparkSession, csv_path: str) -> DataFrame:
    """
    Loads credit card transaction CSV data into a PySpark DataFrame with schema inference.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset path does not exist: {csv_path}")

    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    return df


def inspect_dataset(df: DataFrame, num_records: int = 10) -> Dict[str, Any]:
    """
    Inspects and displays dataset schema, record count, and missing values.
    Returns summary metadata dictionary.
    """
    num_rows = df.count()
    num_cols = len(df.columns)
    dtypes_map = dict(df.dtypes)
    
    print("\n" + "=" * 60)
    print(f" DATASET SCHEMA & OVERVIEW ({num_rows:,} rows, {num_cols} columns)")
    print("=" * 60)
    df.printSchema()

    print(f"\n--- First {num_records} Sample Records ---")
    df.show(num_records, truncate=True)

    # Calculate missing values count per column
    print("\n--- Missing Value Analysis ---")
    missing_counts = {}
    for col_name in df.columns:
        dt = dtypes_map.get(col_name, "")
        if dt in ["double", "float"]:
            null_count = df.filter(F.col(col_name).isNull() | F.isnan(F.col(col_name))).count()
        else:
            null_count = df.filter(F.col(col_name).isNull()).count()
        missing_counts[col_name] = null_count
        if null_count > 0:
            pct = (null_count / num_rows) * 100
            print(f"Column '{col_name}': {null_count:,} missing values ({pct:.2f}%)")

    total_missing = sum(missing_counts.values())
    if total_missing == 0:
        print("No missing or NaN values detected across any column.")

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": df.columns,
        "missing_counts": missing_counts
    }


def remove_duplicates_and_impute(df: DataFrame) -> DataFrame:
    """
    Removes exact duplicate rows and applies appropriate imputation:
    - Numerical columns: imputed with median / mean
    - Categorical columns: imputed with 'Unknown'
    """
    # 1. Deduplication
    initial_count = df.count()
    df = df.dropDuplicates()
    dedup_count = df.count()
    dropped_dupes = initial_count - dedup_count
    if dropped_dupes > 0:
        print(f"Removed {dropped_dupes:,} duplicate transaction rows.")

    # 2. Imputation for numeric and string columns
    dtypes_map = dict(df.dtypes)
    for col_name, dtype in dtypes_map.items():
        if dtype in ["int", "bigint", "double", "float"]:
            if dtype in ["double", "float"]:
                null_count = df.filter(F.col(col_name).isNull() | F.isnan(F.col(col_name))).count()
            else:
                null_count = df.filter(F.col(col_name).isNull()).count()
            if null_count > 0:
                median_val = df.approxQuantile(col_name, [0.5], 0.01)[0]
                df = df.na.fill({col_name: median_val})
        elif dtype == "string":
            df = df.na.fill({col_name: "Unknown"})

    return df


def engineer_features(df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Performs domain feature engineering for credit card fraud detection:
    - Standardizes target column to 'label' (DoubleType: 0.0 or 1.0)
    - Extracts temporal features: trans_hour, trans_day, trans_month, trans_dayofweek, is_night
    - Computes geospatial Haversine distance in km between cardholder and merchant
    - Computes cardholder age from date of birth (dob)
    - Drops personally identifiable information (PII) & high-cardinality IDs
    """
    target_col = detect_target_column(df)
    df = df.withColumn("label", F.col(target_col).cast(DoubleType()))

    # 1. Temporal Feature Extraction
    if "trans_date_trans_time" in df.columns:
        col_type = dict(df.dtypes).get("trans_date_trans_time", "")
        if col_type == "timestamp":
            df = df.withColumn("trans_time_ts", F.col("trans_date_trans_time"))
        else:
            df = df.withColumn("trans_time_ts", F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"))
        
        df = df.withColumn("trans_hour", F.hour("trans_time_ts"))
        df = df.withColumn("trans_day", F.dayofmonth("trans_time_ts"))
        df = df.withColumn("trans_month", F.month("trans_time_ts"))
        df = df.withColumn("trans_dayofweek", F.dayofweek("trans_time_ts"))
        # Night transaction indicator (between 22:00 and 05:00)
        df = df.withColumn(
            "is_night",
            F.when((F.col("trans_hour") >= 22) | (F.col("trans_hour") <= 5), 1.0).otherwise(0.0)
        )

    # 2. Geospatial Haversine Distance (Kilometers)
    if all(col in df.columns for col in ["lat", "long", "merch_lat", "merch_long"]):
        r_earth = 6371.0  # Earth's radius in km
        lat1 = F.radians(F.col("lat"))
        lat2 = F.radians(F.col("merch_lat"))
        dlat = F.radians(F.col("merch_lat") - F.col("lat"))
        dlon = F.radians(F.col("merch_long") - F.col("long"))

        a = (
            F.pow(F.sin(dlat / 2.0), 2) +
            F.cos(lat1) * F.cos(lat2) * F.pow(F.sin(dlon / 2.0), 2)
        )
        c = 2.0 * F.atan2(F.sqrt(a), F.sqrt(1.0 - a))
        distance = r_earth * c
        df = df.withColumn("distance_km", F.round(distance, 2))

    # 3. Cardholder Age Feature
    if "dob" in df.columns and "trans_time_ts" in df.columns:
        df = df.withColumn("dob_date", F.to_date("dob", "yyyy-MM-dd"))
        age_col = F.datediff(F.col("trans_time_ts"), F.col("dob_date")) / 365.25
        df = df.withColumn("age", F.round(age_col.cast(DoubleType()), 1))

    # Ensure amount is double
    if "amt" in df.columns:
        df = df.withColumn("amt", F.col("amt").cast(DoubleType()))

    # Ensure city population is double
    if "city_pop" in df.columns:
        df = df.withColumn("city_pop", F.col("city_pop").cast(DoubleType()))

    # 4. Remove Irrelevant Columns and PII Identifiers to avoid leakage & privacy exposure
    pii_columns = [
        "cc_num", "first", "last", "street", "trans_num", "unix_time",
        "trans_date_trans_time", "trans_time_ts", "dob", "dob_date",
        "job", "city", "state", "zip", "lat", "long", "merch_lat", "merch_long",
        "merchant", target_col
    ]
    cols_to_drop = [c for c in pii_columns if c in df.columns and c != "label"]
    df = df.drop(*cols_to_drop)

    # Determine engineered numerical and categorical feature lists
    available_numeric = [c for c in ["amt", "trans_hour", "trans_day", "trans_month", "trans_dayofweek", "is_night", "distance_km", "age", "city_pop"] if c in df.columns]
    available_categorical = [c for c in ["category", "gender"] if c in df.columns]

    feature_meta = {
        "numeric_cols": available_numeric,
        "categorical_cols": available_categorical,
        "label_col": "label"
    }

    return df, feature_meta


def compute_iqr_threshold(train_df: DataFrame, column: str = "amt", factor: float = 3.0) -> Dict[str, float]:
    """
    Computes IQR boundaries (Q1, Q3, IQR, upper_threshold) strictly on training data
    to prevent data leakage.
    """
    quantiles = train_df.approxQuantile(column, [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    upper_threshold = q3 + (factor * iqr)
    lower_threshold = max(0.0, q1 - (factor * iqr))

    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "upper_threshold": float(upper_threshold),
        "lower_threshold": float(lower_threshold),
        "factor": factor
    }


def apply_outlier_treatment(
    df: DataFrame,
    column: str = "amt",
    iqr_stats: Optional[Dict[str, float]] = None,
    preserve_fraud: bool = True
) -> DataFrame:
    """
    Applies outlier treatment using precomputed IQR statistics:
    - Fraudulent transactions naturally possess high amounts. If preserve_fraud is True,
      capping or filtering only applies to legitimate (label=0) transactions, or caps
      values at the upper threshold rather than deleting records, retaining all fraud signals.
    """
    if iqr_stats is None:
        raise ValueError("iqr_stats must be provided (fit on training data).")

    upper_cap = iqr_stats["upper_threshold"]

    if preserve_fraud:
        # Cap amount for legitimate transactions; preserve actual amount for fraudulent transactions
        df = df.withColumn(
            column,
            F.when((F.col("label") == 0.0) & (F.col(column) > upper_cap), upper_cap)
            .otherwise(F.col(column))
        )
    else:
        # Cap all transactions at upper_cap
        df = df.withColumn(
            column,
            F.when(F.col(column) > upper_cap, upper_cap).otherwise(F.col(column))
        )

    return df


def build_pipeline_stages(
    categorical_cols: List[str],
    numeric_cols: List[str]
) -> Tuple[List[Any], List[str]]:
    """
    Builds the PySpark ML feature transformation stages:
    - StringIndexer for each categorical feature (handleInvalid='keep')
    - OneHotEncoder for indexed categorical features (handleInvalid='keep')
    - VectorAssembler combining numeric and encoded categorical features
    """
    stages = []
    cat_indexed_cols = []
    cat_encoded_cols = []

    for col in categorical_cols:
        idx_col = f"{col}_idx"
        indexer = StringIndexer(inputCol=col, outputCol=idx_col, handleInvalid="keep")
        stages.append(indexer)
        cat_indexed_cols.append(idx_col)

        enc_col = f"{col}_vec"
        encoder = OneHotEncoder(inputCol=idx_col, outputCol=enc_col, handleInvalid="keep")
        stages.append(encoder)
        cat_encoded_cols.append(enc_col)

    all_feature_cols = numeric_cols + cat_encoded_cols
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )
    stages.append(assembler)

    return stages, all_feature_cols


if __name__ == "__main__":
    from src.spark_session import get_spark_session, stop_spark_session

    spark = get_spark_session("PreprocessingTest")
    try:
        csv_file = find_dataset_path()
        print(f"Loading dataset from: {csv_file}")
        raw_df = load_dataset(spark, csv_file)
        summary = inspect_dataset(raw_df, num_records=5)
        clean_df = remove_duplicates_and_impute(raw_df)
        feat_df, meta = engineer_features(clean_df)
        print("\nEngineered DataFrame Schema:")
        feat_df.printSchema()
        print("\nFeature Engineering Metadata:", meta)
    finally:
        stop_spark_session()
