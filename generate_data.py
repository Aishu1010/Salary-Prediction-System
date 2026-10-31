"""
Generate a realistic benchmark dataset for Salary Prediction in India (INR).
Includes realistic correlations between Education, Role, Experience, Location, and Salary.
Includes a small proportion of missing values and duplicates to demonstrate data cleaning.
"""
import numpy as np
import pandas as pd
import os
import sys

# Ensure UTF-8 output handling on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def generate_salary_data(n_samples=3200, random_seed=42):
    np.random.seed(random_seed)
    
    job_titles = [
        "Software Engineer", "Senior Software Engineer", "Data Scientist", 
        "Data Analyst", "Machine Learning Engineer", "DevOps Engineer", 
        "Product Manager", "Full Stack Developer", "Cloud Architect", 
        "Business Analyst", "QA Engineer", "UI/UX Designer"
    ]

    locations = ["Bangalore", "Hyderabad", "Pune", "Mumbai", "Delhi-NCR", "Chennai"]
    education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
    genders = ["Male", "Female", "Other"]

    edu_weights = [0.06, 0.58, 0.28, 0.08]
    gender_weights = [0.53, 0.45, 0.02]
    loc_weights = [0.30, 0.22, 0.16, 0.14, 0.12, 0.06]

    job_base = {
        "Data Analyst": 480000,
        "QA Engineer": 440000,
        "UI/UX Designer": 500000,
        "Business Analyst": 520000,
        "Full Stack Developer": 580000,
        "Software Engineer": 620000,
        "DevOps Engineer": 680000,
        "Data Scientist": 750000,
        "Machine Learning Engineer": 780000,
        "Senior Software Engineer": 1050000,
        "Product Manager": 1150000,
        "Cloud Architect": 1200000
    }

    job_exp_inc = {
        "Data Analyst": 95000,
        "QA Engineer": 85000,
        "UI/UX Designer": 105000,
        "Business Analyst": 110000,
        "Full Stack Developer": 135000,
        "Software Engineer": 145000,
        "DevOps Engineer": 155000,
        "Data Scientist": 180000,
        "Machine Learning Engineer": 190000,
        "Senior Software Engineer": 165000,
        "Product Manager": 185000,
        "Cloud Architect": 195000
    }

    edu_premium = {
        "High School": -90000,
        "Bachelor's": 0,
        "Master's": 160000,
        "PhD": 360000
    }

    loc_premium = {
        "Bangalore": 110000,
        "Mumbai": 95000,
        "Delhi-NCR": 75000,
        "Hyderabad": 55000,
        "Pune": 45000,
        "Chennai": 35000
    }

    data = []

    for _ in range(n_samples):
        job = np.random.choice(job_titles)
        loc = np.random.choice(locations, p=loc_weights)
        edu = np.random.choice(education_levels, p=edu_weights)
        gender = np.random.choice(genders, p=gender_weights)
        
        if job in ["Senior Software Engineer", "Cloud Architect", "Product Manager"]:
            exp = round(float(np.random.gamma(shape=5.0, scale=1.4)), 1)
            exp = max(4.0, min(exp, 28.0))
            age = int(25 + exp + np.random.randint(0, 4))
        else:
            exp = round(float(np.random.gamma(shape=2.5, scale=1.8)), 1)
            exp = max(0.0, min(exp, 25.0))
            min_age = 21 if edu != "PhD" else 26
            age = int(min_age + exp + np.random.randint(0, 4))
            
        base = job_base[job]
        inc = job_exp_inc[job]
        e_prem = edu_premium[edu]
        l_prem = loc_premium[loc]
        
        noise = float(np.random.normal(0, 65000))
        salary = base + (exp * inc) + e_prem + l_prem + (age * 11000) + noise
        salary = max(320000, round(salary, -2))
        
        data.append({
            "Age": age,
            "Gender": gender,
            "Education_Level": edu,
            "Job_Title": job,
            "Years_of_Experience": exp,
            "Location": loc,
            "Annual_Salary": salary
        })

    df = pd.DataFrame(data)

    # Introduce a few duplicate rows to demonstrate deduplication in pipeline
    duplicates = df.sample(n=15, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    # Introduce a small number of realistic missing values to test imputation
    nan_indices = np.random.choice(df.index, size=35, replace=False)
    df.loc[nan_indices[:15], "Years_of_Experience"] = np.nan
    df.loc[nan_indices[15:25], "Education_Level"] = np.nan
    df.loc[nan_indices[25:35], "Age"] = np.nan

    csv_path = "salary_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {csv_path} with {len(df)} records.")
    print("Columns:", list(df.columns))
    print("Missing values per column:\n", df.isnull().sum())
    print("\nSalary statistics (in INR ₹):")
    print(df["Annual_Salary"].describe().apply(lambda x: f"{x:,.2f}"))

if __name__ == "__main__":
    generate_salary_data()
