"""
Root-level proxy for model evaluation module.
Allows direct execution or imports from the root directory.
"""
from src.evaluation import (
    evaluate_model_predictions,
    extract_feature_importances,
    explain_precision_recall_tradeoff,
    save_evaluation_artifacts,
)

if __name__ == "__main__":
    import os
    from src.spark_session import get_spark_session, stop_spark_session
    from src.model_training import load_saved_model
    from src.preprocessing import find_dataset_path, load_dataset, remove_duplicates_and_impute, engineer_features

    spark = get_spark_session("EvaluationCLI")
    try:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "decision_tree_model"))
        if not os.path.exists(model_path):
            print(f"[ERROR] Trained model not found at: {model_path}. Please run model_training.py first.")
        else:
            model = load_saved_model(model_path)
            csv_path = find_dataset_path()
            df = load_dataset(spark, csv_path)
            df_clean = remove_duplicates_and_impute(df)
            df_feat, meta = engineer_features(df_clean)

            _, eval_res = evaluate_model_predictions(model, df_feat)
            fi_df = extract_feature_importances(model, meta)
            print("\nEvaluation Tradeoff Summary:")
            print(explain_precision_recall_tradeoff())
    finally:
        stop_spark_session()
