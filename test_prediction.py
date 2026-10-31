import joblib
import pandas as pd
import sys
from utils.data_preprocessing import format_inr, format_monthly_inr

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

model = joblib.load("salary_model.pkl")
preprocessor = joblib.load("artifacts/preprocessing_pipeline.pkl")

# Test 1: Standard input
sample = pd.DataFrame([{
    "Age": 28,
    "Education_Level": "Master's",
    "Job_Title": "Software Engineer",
    "Years_of_Experience": 5.0,
    "Location": "Bangalore"
}])

X_trans = preprocessor.transform(sample)
pred = model.predict(X_trans)[0]
print(f"Test 1 Standard Prediction: {format_inr(pred)} | Monthly: {format_monthly_inr(pred)}")

# Test 2: Unseen category safety
unseen_sample = pd.DataFrame([{
    "Age": 32,
    "Education_Level": "Doctorate",
    "Job_Title": "Quantum Architect",
    "Years_of_Experience": 8.0,
    "Location": "Tokyo"
}])
X_unseen_trans = preprocessor.transform(unseen_sample)
pred_unseen = model.predict(X_unseen_trans)[0]
print(f"Test 2 Unseen Category Prediction: {format_inr(pred_unseen)} | Monthly: {format_monthly_inr(pred_unseen)}")

