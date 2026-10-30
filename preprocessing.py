"""
Root-level proxy for data preprocessing module.
Allows direct execution or imports from the root directory.
"""
from src.preprocessing import (
    find_dataset_path,
    detect_target_column,
    load_dataset,
    inspect_dataset,
    remove_duplicates_and_impute,
    engineer_features,
    compute_iqr_threshold,
    apply_outlier_treatment,
    build_pipeline_stages,
)

if __name__ == "__main__":
    from src.spark_session import get_spark_session, stop_spark_session

    spark = get_spark_session("PreprocessingCLI")
    try:
        csv_file = find_dataset_path()
        print(f"Loading dataset from: {csv_file}")
        raw_df = load_dataset(spark, csv_file)
        inspect_dataset(raw_df, num_records=5)
        clean_df = remove_duplicates_and_impute(raw_df)
        feat_df, meta = engineer_features(clean_df)
        print("\n[SUCCESS] Preprocessing completed.")
    finally:
        stop_spark_session()
