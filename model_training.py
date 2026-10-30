"""
Root-level proxy for model training module.
Allows direct execution or imports from the root directory.
"""
from src.model_training import (
    stratified_train_test_split,
    add_class_weights,
    balance_training_data,
    train_decision_tree,
    save_trained_model,
    load_saved_model,
)

if __name__ == "__main__":
    import os
    from src.spark_session import get_spark_session, stop_spark_session
    from src.preprocessing import (
        find_dataset_path,
        load_dataset,
        remove_duplicates_and_impute,
        engineer_features,
        compute_iqr_threshold,
        apply_outlier_treatment,
    )
    from pyspark.sql import functions as F

    spark = get_spark_session("ModelTrainingCLI")
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
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "decision_tree_model"))
        save_trained_model(model, save_path)
        print("[SUCCESS] Model trained and persisted.")
    finally:
        stop_spark_session()
