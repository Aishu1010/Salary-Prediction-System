"""
Script to verify that all computations, charts, and data filters in app.py
execute without any exceptions or errors.
"""
import json
import joblib
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from utils.data_preprocessing import (
    load_raw_data,
    clean_data,
    format_inr,
    format_monthly_inr,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLS,
    TARGET_COL,
    EDUCATION_LEVELS
)

def test_full_pipeline():
    print("[1] Testing data loading & cleaning...")
    raw_df = load_raw_data("salary_dataset.csv")
    df, stats = clean_data(raw_df)
    assert len(df) > 3000
    print(f"    Loaded {len(df)} records successfully.")

    print("[2] Testing model and preprocessor loading...")
    model = joblib.load("salary_model.pkl")
    preprocessor = joblib.load("artifacts/preprocessing_pipeline.pkl")
    with open("artifacts/model_metadata.json", "r") as f:
        metadata = json.load(f)
    print("    Model, preprocessor, and metadata loaded successfully.")

    print("[3] Testing salary prediction across multiple roles...")
    test_cases = [
        {"Age": 24, "Years_of_Experience": 2.0, "Education_Level": "Bachelor's", "Job_Title": "Data Analyst", "Location": "Bangalore"},
        {"Age": 30, "Years_of_Experience": 7.0, "Education_Level": "Master's", "Job_Title": "Data Scientist", "Location": "Hyderabad"},
        {"Age": 38, "Years_of_Experience": 14.0, "Education_Level": "PhD", "Job_Title": "Cloud Architect", "Location": "Mumbai"},
        {"Age": 22, "Years_of_Experience": 0.5, "Education_Level": "High School", "Job_Title": "QA Engineer", "Location": "Pune"},
    ]
    for case in test_cases:
        df_in = pd.DataFrame([case])
        x_trans = preprocessor.transform(df_in)
        pred = model.predict(x_trans)[0]
        formatted = format_inr(pred)
        monthly = format_monthly_inr(pred)
        print(f"    {case['Job_Title']} ({case['Years_of_Experience']} yrs exp): {formatted} | {monthly}")
        assert pred > 300000

    print("[4] Testing all EDA Plotly chart generations...")
    # Chart 1: scatter
    fig_scatter = px.scatter(df, x="Years_of_Experience", y="Annual_Salary", color="Education_Level")
    # Chart 2: histogram
    fig_hist = px.histogram(df, x="Annual_Salary", color="Education_Level")
    # Chart 3: job bar
    job_avg = df.groupby("Job_Title")["Annual_Salary"].mean().sort_values().reset_index()
    fig_job = px.bar(job_avg, x="Annual_Salary", y="Job_Title", orientation="h")
    # Chart 4: loc bar
    loc_avg = df.groupby("Location")["Annual_Salary"].mean().reset_index()
    fig_loc = px.bar(loc_avg, x="Location", y="Annual_Salary")
    # Chart 5: box
    fig_box = px.box(df, x="Education_Level", y="Annual_Salary")
    # Chart 6: corr
    corr = df[["Age", "Years_of_Experience", "Annual_Salary"]].dropna().corr()
    fig_corr = px.imshow(corr)
    print("    All 6 EDA figures generated without errors.")

    print("[5] Testing Test Results artifacts...")
    test_res = pd.read_csv("artifacts/test_results.csv")
    assert "Actual_Salary" in test_res.columns
    assert "LR_Predicted_Salary" in test_res.columns
    assert "LR_Residual" in test_res.columns
    print(f"    Test results verified ({len(test_res)} samples).")

    print("[6] Testing Dataset Explorer filtering logic...")
    filtered = df[(df["Years_of_Experience"] >= 2.0) & (df["Years_of_Experience"] <= 8.0)]
    assert len(filtered) > 0
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 0
    print("    Filtering and CSV generation verified.")

    print("\n>>> ALL CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_full_pipeline()
