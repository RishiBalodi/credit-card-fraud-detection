import os
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

def load_data(spark, csv_path):
    """
    Loads credit card transaction data from CSV into a PySpark DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    return df

def engineer_features(df):
    """
    Engineers spending behavior and contextual features from raw transaction fields:
    - trans_hour: Hour of the day (0-23)
    - trans_dayofweek: Day of the week (1=Sunday, 7=Saturday)
    - is_night: Binary flag indicating nighttime transaction (22:00 to 05:00)
    - distance_km: Physical distance between cardholder and merchant
    - age: Cardholder age at transaction time
    """
    # 1. Date & Time features
    df = df.withColumn("trans_time_ts", F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"))
    df = df.withColumn("trans_hour", F.hour("trans_time_ts"))
    df = df.withColumn("trans_dayofweek", F.dayofweek("trans_time_ts"))
    df = df.withColumn(
        "is_night",
        F.when((F.col("trans_hour") >= 22) | (F.col("trans_hour") <= 5), 1.0).otherwise(0.0)
    )

    # 2. Distance in Kilometers (Haversine formula approximation)
    # Radius of earth in km ~ 6371.0
    r_earth = 6371.0
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

    # 3. Cardholder Age
    df = df.withColumn("dob_date", F.to_date("dob", "yyyy-MM-dd"))
    age_col = F.datediff(F.col("trans_time_ts"), F.col("dob_date")) / 365.25
    df = df.withColumn("age", F.round(age_col.cast(DoubleType()), 1))

    # Ensure amount and label are double type
    df = df.withColumn("amt", F.col("amt").cast(DoubleType()))
    df = df.withColumn("label", F.col("is_fraud").cast(DoubleType()))

    return df

def detect_and_remove_outliers(df, column="amt", method="iqr", factor=3.0, preserve_fraud=True):
    """
    Identifies and removes outliers from numeric spending features using PySpark.
    
    Parameters:
    - df: PySpark DataFrame
    - column: Column name to evaluate (default 'amt')
    - method: 'iqr' (Interquartile Range) or 'percentile' (e.g. 99.5th percentile)
    - factor: IQR multiplier (e.g. 1.5, 3.0) or percentile threshold (e.g. 0.995)
    - preserve_fraud: If True, outlier removal specifically targets non-fraud transactions,
                      ensuring genuine high-value fraud patterns are retained.
                      
    Returns:
    - (cleaned_df, stats_dict)
    """
    total_before = df.count()
    fraud_before = df.filter(F.col("label") == 1.0).count()

    if method == "iqr":
        quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1
        lower_bound = max(0.0, q1 - factor * iqr)
        upper_bound = q3 + factor * iqr
    elif method == "percentile":
        pct = factor if factor < 1.0 else factor / 100.0
        upper_bound = df.approxQuantile(column, [pct], 0.01)[0]
        lower_bound = 0.0
        q1, q3, iqr = 0.0, 0.0, 0.0
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    # Condition for outlier identification
    is_outlier_cond = (F.col(column) > upper_bound) | (F.col(column) < lower_bound)
    
    if preserve_fraud:
        # Remove outliers only from non-fraud class to preserve fraud anomalies
        remove_cond = is_outlier_cond & (F.col("label") == 0.0)
    else:
        remove_cond = is_outlier_cond

    outlier_df = df.filter(remove_cond)
    outliers_count = outlier_df.count()
    
    cleaned_df = df.filter(~remove_cond)
    total_after = cleaned_df.count()
    fraud_after = cleaned_df.filter(F.col("label") == 1.0).count()

    stats = {
        "method": method,
        "factor": factor,
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "total_before": total_before,
        "total_after": total_after,
        "outliers_removed": outliers_count,
        "percentage_removed": round((outliers_count / total_before) * 100, 2) if total_before > 0 else 0.0,
        "fraud_before": fraud_before,
        "fraud_after": fraud_after,
        "fraud_preserved_pct": round((fraud_after / fraud_before) * 100, 2) if fraud_before > 0 else 100.0
    }

    return cleaned_df, stats
