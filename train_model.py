"""
Machine Learning Training Pipeline for Salary Prediction System
--------------------------------------------------------------
Executes the full end-to-end ML workflow:
1. Loads and inspects raw dataset.
2. Cleans data and removes duplicates/anomalies.
3. Splits into training and test sets (prevents data leakage).
4. Fits ColumnTransformer preprocessor on X_train.
5. Trains Multiple Linear Regression (primary algorithm).
6. Trains Random Forest Regressor (comparative model).
7. Evaluates MAE, MSE, RMSE, and R2 score.
8. Persists model, preprocessor, metadata, and test evaluation artifacts.
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure clean UTF-8 console output
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from utils.data_preprocessing import (
    load_raw_data,
    clean_data,
    build_preprocessor,
    format_inr,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLS,
    TARGET_COL,
)


def train_and_evaluate():
    print("=" * 70)
    print("  SALARY PREDICTION SYSTEM - MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load raw dataset
    data_path = "salary_dataset.csv"
    print(f"\n[Step 1] Loading dataset from: {data_path}")
    raw_df = load_raw_data(data_path)
    print(f"Raw dataset shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # 2. Display dataset inspection
    print("\n[Step 2] Dataset Inspection:")
    print("Column names and data types:")
    for col, dtype in raw_df.dtypes.items():
        print(f"  - {col:22} : {dtype}")
    print("\nMissing values before cleaning:")
    print(raw_df.isnull().sum())

    # 3. Clean data
    print("\n[Step 3] Data Cleaning & Anomaly Handling:")
    df_clean, stats = clean_data(raw_df)
    print(f"  - Initial records       : {stats['initial_rows']}")
    print(f"  - Duplicates removed    : {stats['duplicates_removed']}")
    print(f"  - Targets dropped       : {stats.get('invalid_targets_removed', 0)}")
    print(f"  - Anomalies dropped     : {stats.get('logical_anomalies_removed', 0)}")
    print(f"  - Final clean records   : {stats['final_rows']}")

    # 4. Feature and target separation (Strictly exclude Gender from features)
    print(f"\n[Step 4] Feature Selection (Fairness Rule):")
    print(f"  - Predictor Features    : {ALL_FEATURE_COLS}")
    print(f"  - Excluded Non-Job Cols : ['Gender'] (Retained for fairness EDA only)")
    print(f"  - Target Variable       : {TARGET_COL}")

    X = df_clean[ALL_FEATURE_COLS].copy()
    y = df_clean[TARGET_COL].values

    # 5. Train-Test Split (80% Train, 20% Test)
    print("\n[Step 5] Splitting dataset (80% Train, 20% Test, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape : {X_test.shape}")

    # 6. Fit Preprocessor ONLY on training set to avoid data leakage
    print("\n[Step 6] Fitting Preprocessing Pipeline on X_train (Leakage Prevention)...")
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names
    print(f"  - Total one-hot transformed features: {len(all_feature_names)}")

    # 7. Train Primary Model: Multiple Linear Regression
    print("\n[Step 7] Training Multiple Linear Regression (Primary Algorithm)...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_transformed, y_train)
    print("  Linear Regression trained successfully.")

    # 8. Train Comparative Model: Random Forest Regressor
    print("\n[Step 8] Training Random Forest Regressor (Comparative Model)...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_transformed, y_train)
    print("  Random Forest trained successfully.")

    # 9. Evaluate Models
    print("\n[Step 9] Model Evaluation on Test Set:")
    
    # Linear Regression predictions & metrics
    lr_y_pred = lr_model.predict(X_test_transformed)
    lr_mae = mean_absolute_error(y_test, lr_y_pred)
    lr_mse = mean_squared_error(y_test, lr_y_pred)
    lr_rmse = float(np.sqrt(lr_mse))
    lr_r2 = r2_score(y_test, lr_y_pred)
    lr_residuals = y_test - lr_y_pred
    residual_std = float(np.std(lr_residuals))

    # Random Forest predictions & metrics
    rf_y_pred = rf_model.predict(X_test_transformed)
    rf_mae = mean_absolute_error(y_test, rf_y_pred)
    rf_mse = mean_squared_error(y_test, rf_y_pred)
    rf_rmse = float(np.sqrt(rf_mse))
    rf_r2 = r2_score(y_test, rf_y_pred)

    print("\n" + "=" * 65)
    print(f"{'Metric':<25} | {'Linear Regression (Primary)':<20} | {'Random Forest':<15}")
    print("-" * 65)
    print(f"{'MAE (Mean Abs Error)':<25} | {format_inr(lr_mae, False):<20} | {format_inr(rf_mae, False):<15}")
    print(f"{'RMSE (Root Mean Sq Error)':<25} | {format_inr(lr_rmse, False):<20} | {format_inr(rf_rmse, False):<15}")
    print(f"{'MSE (Mean Sq Error)':<25} | {lr_mse:>19,.0f} | {rf_mse:>14,.0f}")
    print(f"{'R-squared (R2) Score':<25} | {lr_r2:>19.4f} | {rf_r2:>14.4f}")
    print("=" * 65)

    # 10. Sample Comparison: Actual vs Predicted
    print("\n[Step 10] Comparison of Actual vs Predicted Salaries (First 5 Test Samples):")
    sample_df = X_test.head(5).copy()
    sample_df["Actual_Salary"] = [format_inr(val) for val in y_test[:5]]
    sample_df["LR_Predicted"] = [format_inr(val) for val in lr_y_pred[:5]]
    sample_df["Difference"] = [format_inr(y_test[i] - lr_y_pred[i], False) for i in range(5)]
    print(sample_df[["Job_Title", "Years_of_Experience", "Actual_Salary", "LR_Predicted", "Difference"]].to_string())

    # 11. Feature Coefficients / Importance
    coef_dict = dict(zip(all_feature_names, lr_model.coef_))
    sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 8 Feature Coefficients (Multiple Linear Regression):")
    for feat, coef in sorted_coefs[:8]:
        direction = "+" if coef >= 0 else "-"
        print(f"  {feat:35} : {direction} {format_inr(abs(coef), False)}")

    # 12. Save Models and Preprocessing Artifacts
    print("\n[Step 11] Persisting Models and Artifacts using Joblib & JSON...")
    
    # Save primary model and preprocessor
    joblib.dump(lr_model, "salary_model.pkl")
    joblib.dump(rf_model, "artifacts/random_forest_model.pkl")
    joblib.dump(preprocessor, "artifacts/preprocessing_pipeline.pkl")
    print("  Saved: salary_model.pkl")
    print("  Saved: artifacts/random_forest_model.pkl")
    print("  Saved: artifacts/preprocessing_pipeline.pkl")

    # Save detailed evaluation and metadata JSON
    unique_categories = {
        "Job_Title": sorted(df_clean["Job_Title"].dropna().unique().tolist()),
        "Education_Level": ["High School", "Bachelor's", "Master's", "PhD"],
        "Location": sorted(df_clean["Location"].dropna().unique().tolist()),
        "Age_min": int(df_clean["Age"].min()),
        "Age_max": int(df_clean["Age"].max()),
        "Experience_min": float(df_clean["Years_of_Experience"].min()),
        "Experience_max": float(df_clean["Years_of_Experience"].max()),
    }

    metadata = {
        "model_name": "Multiple Linear Regression",
        "comparative_model": "Random Forest Regressor",
        "target": TARGET_COL,
        "features": ALL_FEATURE_COLS,
        "transformed_feature_names": all_feature_names,
        "intercept": float(lr_model.intercept_),
        "coefficients": {k: float(v) for k, v in coef_dict.items()},
        "top_coefficients": [
            {"feature": f, "coefficient": float(c)} for f, c in sorted_coefs
        ],
        "metrics": {
            "linear_regression": {
                "mae": float(lr_mae),
                "mse": float(lr_mse),
                "rmse": float(lr_rmse),
                "r2": float(lr_r2),
                "residual_std": float(residual_std),
            },
            "random_forest": {
                "mae": float(rf_mae),
                "mse": float(rf_mse),
                "rmse": float(rf_rmse),
                "r2": float(rf_r2),
            },
        },
        "dataset_summary": {
            "total_records": int(len(df_clean)),
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "avg_salary": float(df_clean[TARGET_COL].mean()),
            "median_salary": float(df_clean[TARGET_COL].median()),
            "max_salary": float(df_clean[TARGET_COL].max()),
            "min_salary": float(df_clean[TARGET_COL].min()),
            "avg_experience": float(df_clean["Years_of_Experience"].mean()),
        },
        "categories": unique_categories,
    }

    with open("artifacts/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("  Saved: artifacts/model_metadata.json")

    # Save test set evaluation DataFrame for interactive plots in Streamlit
    test_eval_df = X_test.copy()
    test_eval_df["Actual_Salary"] = y_test
    test_eval_df["LR_Predicted_Salary"] = lr_y_pred
    test_eval_df["RF_Predicted_Salary"] = rf_y_pred
    test_eval_df["LR_Residual"] = lr_residuals
    test_eval_df.to_csv("artifacts/test_results.csv", index=False)
    print("  Saved: artifacts/test_results.csv")

    print("\nTraining workflow completed successfully!")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()

