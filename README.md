# 💼 Salary Prediction System Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Currency](https://img.shields.io/badge/Currency-INR%20%28%E2%82%B9%29-10B981)

An end-to-end Machine Learning web application designed to predict and analyze expected annual and monthly compensation for tech industry professionals in India. Powered by **Multiple Linear Regression** with **Random Forest Regressor** benchmarking, this system features a responsive, multi-page **Streamlit** dashboard with interactive **Plotly** visualizations, comprehensive exploratory data analysis, and ethical AI auditing.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Objectives](#-objectives)
4. [Technologies Used](#-technologies-used)
5. [Dataset Information & Labeling](#-dataset-information--labeling)
6. [Machine Learning Workflow](#-machine-learning-workflow)
7. [Model Evaluation & Benchmarking](#-model-evaluation--benchmarking)
8. [Installation & Setup](#-installation--setup)
9. [How to Run the Application](#-how-to-run-the-application)
10. [Example Prediction Walkthrough](#-example-prediction-walkthrough)
11. [Streamlit Dashboard Pages & Screenshots](#-streamlit-dashboard-pages--screenshots)
12. [Ethical AI & Fairness Guarantees](#-ethical-ai--fairness-guarantees)
13. [Future Improvements](#-future-improvements)

---

## 📖 Project Overview
Compensation transparency is a major challenge for job seekers, human resource professionals, and university graduates entering the competitive technology workforce. The **Salary Prediction System** models the multi-factor relationships between a candidate's formal education level, years of experience, specific job role, age, and metropolitan employment location to generate realistic, data-driven salary estimates in Indian Rupees (INR ₹).

The application is structured as a five-page interactive dashboard suitable for academic demonstration, college AI/Data Science capstone portfolios, and practical salary benchmarking.

---

## 🎯 Problem Statement
Salaries across technical roles vary widely depending on experience, job title demand, educational attainment, and geographic location. Candidates often lack reliable estimates when negotiating compensation or planning career trajectories, while recruiters spend considerable effort calibrating offers. 

**Core Question:** *How accurately can an employee's annual compensation be modeled using observable, merit-based qualifications while eliminating non-job demographic bias?*

---

## 🚀 Objectives
- Build an end-to-end Machine Learning pipeline using **Multiple Linear Regression** as the primary algorithm.
- Implement robust preprocessing using Scikit-Learn's `ColumnTransformer` and `Pipeline` to prevent data leakage.
- Safely handle missing values, duplicates, and unseen categorical inputs.
- Train a comparative non-linear benchmark model (**Random Forest Regressor**) to evaluate model behavior.
- Design a modern, responsive, 5-page **Streamlit** dashboard displaying interactive Plotly charts, key performance metrics, factor breakdowns, confidence intervals, and a searchable dataset explorer with CSV export.
- Strictly adhere to AI fairness guidelines by excluding demographic attributes (e.g., gender) from prediction features.

---

## 💻 Technologies Used
| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming language |
| **Data Processing** | Pandas, NumPy | Data cleaning, matrix manipulation, aggregations |
| **Machine Learning** | Scikit-Learn | Preprocessing pipelines, Linear Regression, Random Forest |
| **Persistence** | Joblib | Serialization of trained models and preprocessing artifacts |
| **Data Visualization** | Plotly Express & Graph Objects | Modern interactive charts, scatter plots, boxplots, heatmaps |
| **Visual Styling** | Matplotlib, Seaborn | Static exploratory analysis and distribution checking |
| **Web Interface** | Streamlit | Modern reactive dashboard and UI controls |

---

## 📊 Dataset Information & Labeling
- **Type:** **Realistic Synthetic Benchmark Dataset** (explicitly labeled for academic and demonstration purposes).
- **Domain:** Indian Information Technology (IT) & Software Engineering compensation benchmarks across Tier-1 and Tier-2 metropolitan hubs.
- **Records:** 3,200+ samples.
- **Currency:** Indian Rupee (INR ₹), formatted with Lakhs Per Annum (LPA) standards.

### Column Schema
| Column Name | Data Type | Description | Role in ML Pipeline |
| :--- | :--- | :--- | :--- |
| `Age` | Float / Int | Candidate age (18 to 65) | Predictor Feature |
| `Gender` | Object | Male / Female / Other | **Excluded from ML Model** (Retained for fairness audit only) |
| `Education_Level` | Object | High School, Bachelor's, Master's, PhD | Categorical Feature |
| `Job_Title` | Object | 12 Roles (Software Engineer, Data Scientist, Cloud Architect, etc.) | Categorical Feature |
| `Years_of_Experience` | Float | Relevant work experience (0.0 to 28+ years) | Numerical Feature |
| `Location` | Object | Bangalore, Hyderabad, Pune, Mumbai, Delhi-NCR, Chennai | Categorical Feature |
| `Annual_Salary` | Float | Annual compensation in INR (₹ 3,20,000 to ₹ 61,80,000+) | Target Variable ($y$) |

---

## ⚙️ Machine Learning Workflow

```
┌────────────────────────┐
│   salary_dataset.csv   │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Data Ingestion & Cleaning (utils/data_preprocessing)   │
│  - Deduplication & logic validation (Age ≥ Exp + 16)   │
│  - Split predictor features (X) vs Target (y)          │
│  - Gender isolated from feature set                    │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Train-Test Split (80% Train, 20% Test, random_state)  │
│  Prevents data leakage before any scaling/encoding    │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Fitted ColumnTransformer (artifacts/pipeline.pkl)     │
│  - Numerical: SimpleImputer(median) + StandardScaler   │
│  - Categorical: SimpleImputer(mode) + OneHotEncoder    │
│    (handle_unknown='ignore')                           │
└───────────┬────────────────────────────────────────────┘
            │
      ┌─────┴──────────────────┐
      ▼                        ▼
┌────────────────────┐   ┌───────────────────────────┐
│ Multiple Linear    │   │ Random Forest Regressor   │
│ Regression         │   │ (Comparative Benchmark)   │
│ (Primary Model)    │   │                           │
└─────┬──────────────┘   └─────┬─────────────────────┘
      │                        │
      └───────────┬────────────┘
                  ▼
┌────────────────────────────────────────────────────────┐
│  Model Evaluation (artifacts/model_metadata.json)      │
│  MAE, MSE, RMSE, R² Score, Residual Diagnostics       │
└────────────────────────────────────────────────────────┘
```

1. **Data Ingestion & Cleaning:** Reads `salary_dataset.csv`, cleans missing entries via statistical imputation, validates age-experience sanity constraints, and strips trailing whitespaces.
2. **Leakage-Free Splitting:** The data is split into 80% training (`N=2,560`) and 20% testing (`N=641`) partitions before any transformation is applied.
3. **Feature Preprocessing:**
   - Numerical columns (`Age`, `Years_of_Experience`) undergo median imputation and Z-score standardization.
   - Categorical columns (`Education_Level`, `Job_Title`, `Location`) are one-hot encoded with `handle_unknown='ignore'` so unseen production inputs never crash the pipeline.
4. **Model Training:**
   - **Primary Model:** Ordinary Least Squares (OLS) Multiple Linear Regression.
   - **Comparative Model:** Random Forest Regressor (100 estimators).
5. **Evaluation & Artifact Storage:** Calculates regression metrics, generates residual diagnostics, and serializes models to `salary_model.pkl` and `artifacts/`.

---

## 📈 Model Evaluation & Benchmarking

Evaluated on the independent 20% test partition (`N = 641 records`):

| Metric | Multiple Linear Regression (Primary) | Random Forest Regressor (Benchmark) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | **₹ 97,234** | ₹ 80,236 | Average magnitude of error in INR |
| **Root Mean Squared Error (RMSE)** | **₹ 1,28,686** | ₹ 1,12,361 | Standard deviation of prediction errors |
| **Mean Squared Error (MSE)** | **1.656 × 10¹⁰** | 1.263 × 10¹⁰ | Variance of squared prediction errors |
| **R² Score (Goodness of Fit)** | **0.9747 (97.5%)** | **0.9807 (98.1%)** | Proportion of salary variance explained |

### Why Multiple Linear Regression is the Primary Model
While ensemble models like Random Forest achieve slightly lower RMSE, **Multiple Linear Regression is chosen as the primary production model** because:
1. **Explainability & Interpretability:** Each feature has a direct, transparent rupee coefficient. Recruiters and students can see exactly how much an extra year of experience (+₹ 4.54L) or a Master's degree contributes to pay.
2. **No Black-Box Obscurity:** Critical for compensation fairness and regulatory scrutiny.
3. **Computational Efficiency:** Sub-millisecond latency for live web queries.

---

## 💻 Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git (optional)

### 2. Clone or Navigate to the Project Directory
```bash
cd "c:\Users\Dell\OneDrive\Desktop\salary prediction"
```

### 3. (Recommended) Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run the Application

### Step 1: Train the Machine Learning Model (Optional if artifacts exist)
```bash
python train_model.py
```
*This command cleans the dataset, trains Multiple Linear Regression and Random Forest models, displays evaluation metrics, and saves `salary_model.pkl` and the `artifacts/` folder.*

### Step 2: Launch the Streamlit Web Application
```bash
streamlit run app.py
```

Streamlit will launch local development server output:
```text
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🔮 Example Prediction Walkthrough

### Candidate Profile Input:
- **Job Title:** Senior Software Engineer
- **Years of Experience:** 8.5 Years
- **Education Level:** Master's Degree
- **Location:** Bangalore
- **Age:** 32 Years

### Output:
- **Predicted Annual Salary:** `₹ 28,45,600 (28.46 LPA)`
- **Estimated Monthly Gross:** `₹ 2,37,133 / month`
- **95% Confidence Interval:** `₹ 25,93,000 — ₹ 30,98,000`
- **Factor Attribution Breakdown:**
  - Experience (+₹ 4.54L scaled per year of expertise)
  - Role baseline premium for Senior Software Engineer
  - Master's degree credential bonus
  - Bangalore metropolitan cost-of-living index

---

## 🖥️ Streamlit Dashboard Pages & Screenshots

The application consists of 5 dedicated dashboard modules:

```
├── 🏠 Page 1: Home Dashboard
│   ├── KPI Metric Cards (Total Records, Mean Salary, Max Salary, Avg Experience)
│   ├── Interactive Salary Distribution Histogram with Median Reference
│   ├── Educational Tier Compensation Chart
│   └── System Architecture & Pipeline Overview
│
├── 🔮 Page 2: Salary Prediction
│   ├── Interactive Profile Form (Role, Experience, Education, Location, Age)
│   ├── Live Input Consistency Validation
│   ├── Annual & Monthly Estimated Compensation in INR (₹)
│   ├── 95% Confidence Interval Band (Residual Standard Error)
│   ├── Comparative Random Forest Estimate
│   └── Factor Decomposition & Ethics Disclaimer
│
├── 📊 Page 3: Data Analysis (EDA)
│   ├── Tab 1: Experience vs. Salary Scatter with OLS Regression Trendline
│   ├── Tab 2: Job Role & Metro Hub Compensation Rankings (Sorted Bar Charts)
│   ├── Tab 3: Pearson Correlation Heatmap & Demographic Fairness Audit
│
├── 🎯 Page 4: Model Performance
│   ├── Metric Cards: MAE, RMSE, MSE, R² Score
│   ├── Comparative Algorithm Performance Table (OLS vs Random Forest)
│   ├── Actual vs. Predicted Diagonal Calibration Chart
│   ├── Residual Error Scatter Plot
│   ├── Top Regression Feature Coefficients
│   └── Beginner's Guide to ML Evaluation
│
└── 📁 Page 5: Dataset Explorer
    ├── Multi-attribute Filter Panel (Role, Location, Degree, Experience, Salary)
    ├── Searchable Data Table with Formatted Rupee Columns
    ├── Statistical Summary Panel
    └── One-Click Filtered CSV Export
```

*(You can capture screenshots of each page directly from your browser at `http://localhost:8501` to add to your project report or submission presentation).*

---

## ⚖️ Ethical AI & Fairness Guarantees
1. **Protected Attributes Excluded from Training:** Demographic variables (e.g., `Gender`) are strictly omitted from the predictor matrix ($X$). Compensation is computed solely based on job duties, verifiable experience, education level, and geographic office tier.
2. **Audit & Transparency:** The dataset retains gender information strictly for the exploratory fairness audit tab in Page 3, enabling transparent evaluation of demographic parity without introducing algorithmic bias into predictions.
3. **Statistical Disclaimer:** Predictions are explicitly communicated as statistical estimations based on historical benchmarks, avoiding misleading accuracy claims or guaranteed contract promises.

---

## 🔮 Future Improvements
- **Integration of Real-Time APIs:** Connect to live job board APIs (e.g., LinkedIn, Indeed) for dynamic salary updates.
- **Skill-Based Feature Engineering:** Incorporate specific technical proficiencies (e.g., Kubernetes, PyTorch, React) into the regression model.
- **Bonus & Equity Modeling:** Expand the target variable to break down base pay, variable cash bonus, and Employee Stock Ownership Plans (ESOPs).
- **Cloud Deployment:** Deploy the application container on Streamlit Cloud, AWS EC2, or Hugging Face Spaces with automated CI/CD.

---

## 👨‍💻 Project Structure
```
salary-prediction/
├── app.py                          # Complete multi-page Streamlit application
├── train_model.py                  # End-to-end ML training and evaluation script
├── generate_data.py                # Realistic synthetic benchmark data generator
├── test_prediction.py              # Standalone inference and edge-case testing script
├── verify_pages.py                 # Automated pipeline and chart verification script
├── requirements.txt                # Pinned project dependencies
├── README.md                       # Comprehensive project documentation
├── salary_dataset.csv              # Benchmark salary dataset in INR (3,200+ rows)
├── salary_model.pkl                # Trained Multiple Linear Regression model
├── artifacts/
│   ├── preprocessing_pipeline.pkl  # Fitted Scikit-Learn ColumnTransformer
│   ├── random_forest_model.pkl     # Comparative Random Forest benchmark model
│   ├── model_metadata.json         # Evaluated metrics, coefficients, category schemas
│   └── test_results.csv            # Test set predictions and residual diagnostics
└── utils/
    ├── __init__.py                 # Utils package initializer
    └── data_preprocessing.py       # Data cleaning, pipeline builders, and INR formatters
```

---
*Created as part of the AI & Data Science Capstone Project Series.*

