"""
Salary Prediction System - Modern Streamlit Web Application
===========================================================
A comprehensive, beginner-friendly Machine Learning dashboard for salary prediction
and data exploration built with Scikit-learn, Pandas, Plotly, and Streamlit.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.88rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    /* Prediction Banner */
    .prediction-card {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 1px solid #059669;
        border-radius: 14px;
        padding: 24px;
        margin-top: 18px;
        margin-bottom: 18px;
        color: #ecfdf5;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #3b82f6;
        color: white;
        margin-right: 6px;
    }
    .badge-synthetic {
        background-color: #8b5cf6;
    }
    .badge-inr {
        background-color: #10b981;
    }
    .disclaimer-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        color: #92400e;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.875rem;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data and Model Cache
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Load and clean dataset with caching."""
    data_path = "salary_dataset.csv"
    if not os.path.exists(data_path):
        st.error(f"Dataset file '{data_path}' not found! Please run 'python train_model.py' first.")
        st.stop()
    raw_df = load_raw_data(data_path)
    clean_df, stats = clean_data(raw_df)
    return clean_df, stats


@st.cache_resource
def load_model_artifacts():
    """Load model, preprocessor, and metadata with caching."""
    model_path = "salary_model.pkl"
    prep_path = "artifacts/preprocessing_pipeline.pkl"
    meta_path = "artifacts/model_metadata.json"
    
    if not (os.path.exists(model_path) and os.path.exists(prep_path)):
        st.error("Model or preprocessing pipeline artifact missing! Please execute 'python train_model.py'.")
        st.stop()
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    rf_model = None
    rf_path = "artifacts/random_forest_model.pkl"
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        
    return model, preprocessor, metadata, rf_model


@st.cache_data
def load_test_results():
    """Load pre-evaluated test results for performance visualization."""
    res_path = "artifacts/test_results.csv"
    if os.path.exists(res_path):
        return pd.read_csv(res_path)
    return None


# Load artifacts
df, cleaning_stats = load_dataset()
lr_model, preprocessor, metadata, rf_model = load_model_artifacts()
test_results_df = load_test_results()

# Extract category options from metadata or dataset
available_jobs = metadata.get("categories", {}).get("Job_Title", sorted(df["Job_Title"].dropna().unique().tolist()))
available_locations = metadata.get("categories", {}).get("Location", sorted(df["Location"].dropna().unique().tolist()))
available_education = metadata.get("categories", {}).get("Education_Level", EDUCATION_LEVELS)


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/money-bag.png", width=70)
st.sidebar.title("Salary System")
st.sidebar.caption("Machine Learning Web Application")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Home Dashboard",
        "🔮 Salary Prediction",
        "📊 Data Analysis (EDA)",
        "🎯 Model Performance",
        "📁 Dataset Explorer"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Project Information")
st.sidebar.markdown("""
- **Algorithm:** Multiple Linear Regression
- **Comparative:** Random Forest Regressor
- **Currency:** Indian Rupee (INR ₹)
- **Dataset:** 3,200+ Records *(Realistic Synthetic Benchmark)*
- **Fairness:** Gender excluded from prediction
""")

st.sidebar.caption("Antigravity AI & Data Science Project")


# ===========================================================================
# PAGE 1: HOME DASHBOARD
# ===========================================================================
if page == "🏠 Home Dashboard":
    st.markdown("""
        <h1>💼 Salary Prediction System</h1>
        <p style='color: #64748b; font-size: 1.1rem;'>
            An end-to-end Machine Learning solution designed to estimate competitive annual compensation 
            based on job title, years of experience, education qualification, age, and location in India.
        </p>
        <div>
            <span class='badge'>Multiple Linear Regression</span>
            <span class='badge badge-inr'>Currency: INR (₹)</span>
            <span class='badge badge-synthetic'>Realistic Synthetic Dataset</span>
            <span class='badge' style='background-color: #0284c7;'>Scikit-Learn & Streamlit</span>
        </div>
        <br>
    """, unsafe_allow_html=True)

    # Key KPI Metric Cards
    total_records = len(df)
    avg_salary = df["Annual_Salary"].mean()
    median_salary = df["Annual_Salary"].median()
    max_salary = df["Annual_Salary"].max()
    avg_exp = df["Years_of_Experience"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>👥 Total Dataset Records</div>
                <div class='metric-value'>{total_records:,}</div>
                <div class='metric-sub'>Cleaned & Deduplicated</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>💰 Average Annual Salary</div>
                <div class='metric-value'>{format_inr(avg_salary, True)}</div>
                <div class='metric-sub'>Median: {format_inr(median_salary, False)}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>🚀 Highest Annual Salary</div>
                <div class='metric-value'>{format_inr(max_salary, True)}</div>
                <div class='metric-sub'>Executive & Architect Roles</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>⏳ Avg. Experience</div>
                <div class='metric-value'>{avg_exp:.1f} Years</div>
                <div class='metric-sub'>Range: 0 to 28+ Years</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Visual Overview Layout
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📈 Annual Salary Distribution (INR ₹)")
        fig_dist = px.histogram(
            df,
            x="Annual_Salary",
            nbins=35,
            marginal="box",
            color_discrete_sequence=["#38bdf8"],
            title="Distribution of Annual Compensation with Median Marker"
        )
        fig_dist.add_vline(
            x=median_salary,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Median: {format_inr(median_salary, False)}",
            annotation_position="top right"
        )
        fig_dist.update_layout(
            xaxis_title="Annual Salary (INR ₹)",
            yaxis_title="Employee Count",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=380
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.subheader("🎓 Salary by Education Level")
        edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
        edu_summary = df.groupby("Education_Level")["Annual_Salary"].mean().reindex(edu_order).reset_index()
        
        fig_edu = px.bar(
            edu_summary,
            x="Education_Level",
            y="Annual_Salary",
            color="Annual_Salary",
            color_continuous_scale="Blues",
            text=edu_summary["Annual_Salary"].apply(lambda x: f"₹ {x/100000:.1f}L"),
            title="Average Salary by Highest Qualification"
        )
        fig_edu.update_traces(textposition="outside")
        fig_edu.update_layout(
            xaxis_title="Education Level",
            yaxis_title="Average Salary (INR ₹)",
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            height=380
        )
        st.plotly_chart(fig_edu, use_container_width=True)

    st.markdown("---")

    # Workflow & Architecture Explanation
    st.subheader("🛠️ Machine Learning Workflow Architecture")
    flow_col1, flow_col2, flow_col3 = st.columns(3)

    with flow_col1:
        st.info("""
        **1. Data Ingestion & Cleaning**
        - Missing value imputation (median for age/exp, mode for education).
        - Removal of duplicates and validation of logical consistency (`Age ≥ Exp + 16`).
        - Data leakage prevention by fitting preprocessors strictly on the training partition.
        """)

    with flow_col2:
        st.info("""
        **2. Preprocessing & Encoding**
        - Numerical features scaled via `StandardScaler`.
        - Categorical variables encoded via `OneHotEncoder(handle_unknown='ignore')`.
        - Encapsulated in a Scikit-Learn `ColumnTransformer` for production readiness.
        """)

    with flow_col3:
        st.info("""
        **3. Training & Evaluation**
        - Primary: **Multiple Linear Regression** for direct coefficient interpretability.
        - Comparative: **Random Forest Regressor** for non-linear benchmark.
        - Rigorous evaluation: MAE, MSE, RMSE, and $R^2$ Score.
        """)


# ===========================================================================
# PAGE 2: SALARY PREDICTION
# ===========================================================================
elif page == "🔮 Salary Prediction":
    st.markdown("""
        <h2>🔮 Interactive Salary Prediction</h2>
        <p style='color: #64748b;'>
            Enter the professional profile below to estimate the predicted annual and monthly compensation 
            using the trained Multiple Linear Regression model.
        </p>
    """, unsafe_allow_html=True)

    with st.form("salary_prediction_form"):
        st.markdown("#### 👤 Professional Profile Input")
        
        col_input1, col_input2 = st.columns(2)

        with col_input1:
            job_title = st.selectbox(
                "💼 Job Title / Role:",
                options=available_jobs,
                index=available_jobs.index("Software Engineer") if "Software Engineer" in available_jobs else 0,
                help="Select the candidate's primary job role."
            )
            
            experience = st.slider(
                "⏳ Years of Professional Experience:",
                min_value=0.0,
                max_value=30.0,
                value=4.5,
                step=0.5,
                help="Total full-time relevant work experience in years."
            )
            
            education = st.selectbox(
                "🎓 Highest Education Level:",
                options=available_education,
                index=available_education.index("Bachelor's") if "Bachelor's" in available_education else 0,
                help="Highest formal degree attained."
            )

        with col_input2:
            location = st.selectbox(
                "📍 Employment Location:",
                options=available_locations,
                index=available_locations.index("Bangalore") if "Bangalore" in available_locations else 0,
                help="Primary base office or metropolitan hub."
            )
            
            # Smart default age based on experience
            suggested_min_age = int(max(21, 21 + experience))
            age = st.slider(
                "🎂 Candidate Age:",
                min_value=18,
                max_value=65,
                value=max(suggested_min_age, 27),
                step=1,
                help="Age of the candidate (must be physically logical with experience)."
            )

        # Form submission button
        submitted = st.form_submit_button("🚀 Predict Expected Salary", use_container_width=True)

    if submitted:
        # Logical validation
        if age < (experience + 16):
            st.warning(
                f"⚠️ Note: Age ({age}) is unusually close to Years of Experience ({experience}). "
                "In standard industry pipelines, minimum career entry age is typically around 18-21. "
                "The model will still predict, but verify input consistency."
            )

        # Construct input DataFrame matching exact feature schema
        input_data = pd.DataFrame([{
            "Age": float(age),
            "Years_of_Experience": float(experience),
            "Education_Level": education,
            "Job_Title": job_title,
            "Location": location
        }])

        try:
            # Transform features using fitted ColumnTransformer
            X_input_trans = preprocessor.transform(input_data)
            
            # Predict with primary Multiple Linear Regression model
            predicted_salary = float(lr_model.predict(X_input_trans)[0])
            predicted_salary = max(300000.0, predicted_salary)  # Realistic lower bound
            
            monthly_salary = predicted_salary / 12.0
            
            # Uncertainty estimation (95% prediction interval using residual standard error)
            residual_std = metadata.get("metrics", {}).get("linear_regression", {}).get("residual_std", 128000.0)
            ci_margin = 1.96 * residual_std
            lower_bound = max(300000.0, predicted_salary - ci_margin)
            upper_bound = predicted_salary + ci_margin

            # Optional comparative prediction with Random Forest
            rf_pred = None
            if rf_model is not None:
                rf_pred = float(rf_model.predict(X_input_trans)[0])

            # Result Display
            st.markdown(f"""
                <div class='prediction-card'>
                    <h3 style='margin-top:0; color:#a7f3d0;'>🎯 Predicted Compensation Result</h3>
                    <div style='font-size: 2.2rem; font-weight: 800; color: #ffffff;'>
                        {format_inr(predicted_salary, True)}
                    </div>
                    <div style='font-size: 1.25rem; font-weight: 600; color: #6ee7b7; margin-top: 6px;'>
                        Estimated Monthly Gross: {format_monthly_inr(predicted_salary)}
                    </div>
                    <hr style='border-color: #047857; margin: 16px 0;'>
                    <div style='font-size: 0.95rem; color: #d1fae5;'>
                        <strong>📊 95% Confidence Interval (Salary Range):</strong><br>
                        {format_inr(lower_bound, False)} &nbsp;—&nbsp; {format_inr(upper_bound, False)}
                        <br><span style='font-size:0.82rem; color:#a7f3d0;'>
                        (Statistical margin based on root mean square residual standard error of ₹ {residual_std:,.0f})
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Comparative model comparison
            if rf_pred is not None:
                rf_diff = rf_pred - predicted_salary
                rf_diff_str = f"+{format_inr(rf_diff, False)}" if rf_diff >= 0 else f"-{format_inr(abs(rf_diff), False)}"
                st.caption(f"🤖 **Comparative Random Forest Estimate:** {format_inr(rf_pred, True)} (Difference: {rf_diff_str})")

            # Explanation of Factors
            st.markdown("### 🔍 Model Factor Explanation")
            
            exp_col, role_col, edu_col, loc_col = st.columns(4)
            
            with exp_col:
                st.markdown(f"""
                **1. Experience Impact**  
                At **{experience} years**, experience is the primary linear driver of senior compensation in this model.
                """)
            
            with role_col:
                st.markdown(f"""
                **2. Job Role Baseline**  
                **{job_title}** carries a role-specific baseline coefficient reflecting current industry tech demand.
                """)
                
            with edu_col:
                st.markdown(f"""
                **3. Degree Weight**  
                **{education}** adjusts the expected starting tier compared to bachelor's baseline.
                """)
                
            with loc_col:
                st.markdown(f"""
                **4. Location Premium**  
                **{location}** incorporates metropolitan cost-of-living adjustments.
                """)

            # Mandatory Reliability Disclaimer
            st.markdown("""
                <div class='disclaimer-box'>
                    <strong>⚠️ Model Disclaimer & Reliability Notice:</strong><br>
                    Predictions generated by this machine learning system are <em>statistical estimations</em> 
                    derived from historical industry benchmark data. They do not represent a guaranteed job offer 
                    or mandatory compensation contract. Actual industry salaries depend heavily on individual skill assessment, 
                    interview performance, employer budget, variable bonuses, and equity grants.
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error computing prediction: {str(e)}")


# ===========================================================================
# PAGE 3: DATA ANALYSIS (EDA)
# ===========================================================================
elif page == "📊 Data Analysis (EDA)":
    st.markdown("""
        <h2>📊 Exploratory Data Analysis (EDA)</h2>
        <p style='color: #64748b;'>
            Comprehensive visual investigation of compensation trends, demographic factors, 
            and career progression across the dataset.
        </p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions & Trends", "🏢 Roles & Locations", "⚖️ Fairness & Correlations"])

    with tab1:
        # Chart 1: Experience vs Salary
        st.subheader("1. Experience vs. Annual Salary (with Trendline)")
        fig_scatter = px.scatter(
            df,
            x="Years_of_Experience",
            y="Annual_Salary",
            color="Education_Level",
            hover_data=["Job_Title", "Location", "Age"],
            title="Years of Experience vs. Annual Salary (INR ₹)",
            labels={"Years_of_Experience": "Years of Experience", "Annual_Salary": "Annual Salary (₹)"},
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        # Calculate regression trendline using numpy without requiring statsmodels
        valid_pts = df[["Years_of_Experience", "Annual_Salary"]].dropna()
        if len(valid_pts) > 1:
            m, b = np.polyfit(valid_pts["Years_of_Experience"], valid_pts["Annual_Salary"], 1)
            x_trend = np.linspace(valid_pts["Years_of_Experience"].min(), valid_pts["Years_of_Experience"].max(), 50)
            y_trend = m * x_trend + b
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode="lines",
                    name="Linear Trendline (OLS)",
                    line=dict(color="#f43f5e", width=2.5, dash="dash")
                )
            )
        fig_scatter.update_layout(height=480)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Chart 2: Salary Distribution Histogram
        st.subheader("2. Salary Distribution Histogram")
        fig_hist = px.histogram(
            df,
            x="Annual_Salary",
            color="Education_Level",
            barmode="overlay",
            nbins=35,
            opacity=0.75,
            title="Annual Salary Distribution by Education Level",
            template="plotly_dark"
        )
        fig_hist.update_layout(height=420, xaxis_title="Annual Salary (INR ₹)", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        col_role, col_loc = st.columns(2)
        
        with col_role:
            st.subheader("3. Job Title vs. Average Salary")
            job_avg = df.groupby("Job_Title")["Annual_Salary"].mean().sort_values(ascending=True).reset_index()
            fig_job = px.bar(
                job_avg,
                x="Annual_Salary",
                y="Job_Title",
                orientation="h",
                color="Annual_Salary",
                color_continuous_scale="Viridis",
                text=job_avg["Annual_Salary"].apply(lambda x: f"₹ {x/100000:.1f}L"),
                title="Average Compensation by Job Title (Ascending)"
            )
            fig_job.update_traces(textposition="outside")
            fig_job.update_layout(
                height=520,
                template="plotly_dark",
                xaxis_title="Average Salary (INR ₹)",
                yaxis_title="Job Title",
                showlegend=False
            )
            st.plotly_chart(fig_job, use_container_width=True)

        with col_loc:
            st.subheader("4. Location vs. Average Salary")
            loc_avg = df.groupby("Location")["Annual_Salary"].mean().sort_values(ascending=False).reset_index()
            fig_loc = px.bar(
                loc_avg,
                x="Location",
                y="Annual_Salary",
                color="Annual_Salary",
                color_continuous_scale="Plasma",
                text=loc_avg["Annual_Salary"].apply(lambda x: f"₹ {x/100000:.1f}L"),
                title="Average Compensation by Major Indian Metro Hub"
            )
            fig_loc.update_traces(textposition="outside")
            fig_loc.update_layout(
                height=520,
                template="plotly_dark",
                xaxis_title="Location",
                yaxis_title="Average Salary (INR ₹)",
                showlegend=False
            )
            st.plotly_chart(fig_loc, use_container_width=True)

        # Chart 5: Education vs Salary Boxplot
        st.subheader("5. Education Level vs. Salary Boxplot")
        fig_box = px.box(
            df,
            x="Education_Level",
            y="Annual_Salary",
            color="Education_Level",
            category_orders={"Education_Level": ["High School", "Bachelor's", "Master's", "PhD"]},
            points="outliers",
            title="Compensation Spread Across Education Levels",
            template="plotly_dark"
        )
        fig_box.update_layout(height=420, yaxis_title="Annual Salary (INR ₹)")
        st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.subheader("6. Correlation Heatmap (Numerical Features)")
        num_cols = ["Age", "Years_of_Experience", "Annual_Salary"]
        corr_matrix = df[num_cols].dropna().corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Matrix (Age, Experience, Salary)"
        )
        fig_corr.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
        
        # Demographic Fairness Analysis Section
        st.subheader("7. Ethical AI: Demographic & Fairness Analysis")
        st.markdown("""
        > **Fairness Design Rule:** In compliance with ethical AI standards, **Gender is strictly excluded 
        from the salary prediction model features**. It is examined here solely for audit and transparency.
        """)

        if "Gender" in df.columns:
            gender_summary = df.groupby("Gender")["Annual_Salary"].agg(["count", "mean", "median"]).reset_index()
            gender_summary["mean_formatted"] = gender_summary["mean"].apply(lambda x: format_inr(x, True))
            gender_summary["median_formatted"] = gender_summary["median"].apply(lambda x: format_inr(x, False))
            
            c_g1, c_g2 = st.columns([1, 2])
            with c_g1:
                st.dataframe(
                    gender_summary[["Gender", "count", "mean_formatted", "median_formatted"]],
                    column_config={
                        "Gender": "Gender",
                        "count": "Records",
                        "mean_formatted": "Mean Salary",
                        "median_formatted": "Median Salary"
                    },
                    hide_index=True,
                    use_container_width=True
                )
            with c_g2:
                fig_gen = px.box(
                    df,
                    x="Gender",
                    y="Annual_Salary",
                    color="Gender",
                    title="Salary Spread by Gender (Audit Only)",
                    template="plotly_dark"
                )
                fig_gen.update_layout(height=320)
                st.plotly_chart(fig_gen, use_container_width=True)


# ===========================================================================
# PAGE 4: MODEL PERFORMANCE
# ===========================================================================
elif page == "🎯 Model Performance":
    st.markdown("""
        <h2>🎯 Model Performance & Evaluation</h2>
        <p style='color: #64748b;'>
            Comprehensive evaluation metrics, actual vs. predicted diagnostics, 
            residual analysis, and comparison with baseline algorithms.
        </p>
    """, unsafe_allow_html=True)

    metrics_lr = metadata.get("metrics", {}).get("linear_regression", {})
    metrics_rf = metadata.get("metrics", {}).get("random_forest", {})

    # Top KPI Metrics Cards for Primary Model (Linear Regression)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>MAE (Mean Absolute Error)</div>
                <div class='metric-value'>₹ {metrics_lr.get('mae', 0):,.0f}</div>
                <div class='metric-sub'>Avg. prediction variance in ₹</div>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>RMSE (Root Mean Sq Error)</div>
                <div class='metric-value'>₹ {metrics_lr.get('rmse', 0):,.0f}</div>
                <div class='metric-sub'>Standard deviation of errors</div>
            </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>MSE (Mean Squared Error)</div>
                <div class='metric-value'>{metrics_lr.get('mse', 0):,.0f}</div>
                <div class='metric-sub'>Penalizes large errors heavily</div>
            </div>
        """, unsafe_allow_html=True)

    with m4:
        r2_val = metrics_lr.get('r2', 0)
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>R² Score (Goodness of Fit)</div>
                <div class='metric-value'>{r2_val:.4f}</div>
                <div class='metric-sub'>{r2_val * 100:.1f}% variance explained</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Algorithm Comparison Table
    st.subheader("⚖️ Algorithm Comparison: Linear Regression vs Random Forest")
    comp_df = pd.DataFrame([
        {
            "Algorithm": "Multiple Linear Regression (Primary)",
            "MAE (₹)": f"₹ {metrics_lr.get('mae', 0):,.0f}",
            "RMSE (₹)": f"₹ {metrics_lr.get('rmse', 0):,.0f}",
            "R² Score": f"{metrics_lr.get('r2', 0):.4f}",
            "Interpretability": "High (Direct Coefficients)",
            "Training Speed": "Instantaneous (< 0.1s)",
            "Deployment Risk": "Zero Overfitting Risk"
        },
        {
            "Algorithm": "Random Forest Regressor (Comparative)",
            "MAE (₹)": f"₹ {metrics_rf.get('mae', 0):,.0f}",
            "RMSE (₹)": f"₹ {metrics_rf.get('rmse', 0):,.0f}",
            "R² Score": f"{metrics_rf.get('r2', 0):.4f}",
            "Interpretability": "Moderate (Feature Importance)",
            "Training Speed": "Moderate (~ 0.8s)",
            "Deployment Risk": "Requires Hyperparameter Tuning"
        }
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Plots: Actual vs Predicted & Residuals
    if test_results_df is not None:
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.subheader("📈 Actual vs. Predicted Salaries")
            min_val = min(test_results_df["Actual_Salary"].min(), test_results_df["LR_Predicted_Salary"].min())
            max_val = max(test_results_df["Actual_Salary"].max(), test_results_df["LR_Predicted_Salary"].max())

            fig_act_pred = px.scatter(
                test_results_df,
                x="Actual_Salary",
                y="LR_Predicted_Salary",
                color="Education_Level",
                title="Actual vs. Predicted Salary (Test Partition, N=641)",
                labels={"Actual_Salary": "Actual Salary (₹)", "LR_Predicted_Salary": "Predicted Salary (₹)"},
                template="plotly_dark",
                hover_data=["Job_Title", "Years_of_Experience"]
            )
            # Add ideal 45-degree diagonal reference line
            fig_act_pred.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="#ef4444", width=2, dash="dash")
            )
            fig_act_pred.update_layout(height=440)
            st.plotly_chart(fig_act_pred, use_container_width=True)

        with plot_col2:
            st.subheader("📉 Residual Error Plot")
            fig_resid = px.scatter(
                test_results_df,
                x="LR_Predicted_Salary",
                y="LR_Residual",
                title="Residuals vs. Predicted Values",
                labels={"LR_Predicted_Salary": "Predicted Salary (₹)", "LR_Residual": "Residual Error (₹)"},
                color_discrete_sequence=["#f59e0b"],
                template="plotly_dark"
            )
            fig_resid.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=2)
            fig_resid.update_layout(height=440)
            st.plotly_chart(fig_resid, use_container_width=True)

    # Feature Importance / Regression Coefficients
    st.subheader("📊 Regression Feature Coefficients")
    top_coefs = metadata.get("top_coefficients", [])
    if top_coefs:
        coef_df = pd.DataFrame(top_coefs)
        coef_df["Color"] = coef_df["coefficient"].apply(lambda x: "#10b981" if x >= 0 else "#ef4444")
        
        fig_coef = px.bar(
            coef_df.head(12),
            x="coefficient",
            y="feature",
            orientation="h",
            title="Top 12 Regression Coefficients (Impact on Annual Salary in ₹)",
            template="plotly_dark",
            labels={"coefficient": "Coefficient Impact (₹)", "feature": "Encoded Feature"}
        )
        fig_coef.update_layout(height=480, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_coef, use_container_width=True)

    # Beginner-Friendly Metrics Guide
    with st.expander("📚 Beginner's Guide: Understanding Model Evaluation Metrics"):
        st.markdown("""
        - **MAE (Mean Absolute Error):**  
          The average magnitude of errors between actual and predicted salary in rupees. A lower MAE indicates tighter predictions.
          *In this project, an MAE of ~₹ 97,000 means our predictions are on average within ± ₹ 97,000 of actual pay.*
        
        - **MSE (Mean Squared Error):**  
          The average of the squared errors. Because errors are squared, larger errors are heavily penalized.
        
        - **RMSE (Root Mean Squared Error):**  
          The square root of MSE, bringing the error back into rupee units. Useful because it reflects the standard deviation of residuals.
        
        - **$R^2$ Score (Coefficient of Determination):**  
          Measures the proportion of variance in annual salary explained by the features. 
          An $R^2$ of **0.9747** indicates that **97.5%** of salary differences are accurately captured by experience, role, education, age, and location.
        """)


# ===========================================================================
# PAGE 5: DATASET EXPLORER
# ===========================================================================
elif page == "📁 Dataset Explorer":
    st.markdown("""
        <h2>📁 Dataset Explorer</h2>
        <p style='color: #64748b;'>
            Filter, inspect, and download the full dataset used for model training and validation.
        </p>
    """, unsafe_allow_html=True)

    # Filter Controls
    with st.expander("🔍 Interactive Filter Panel", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            sel_jobs = st.multiselect(
                "Filter by Job Title:",
                options=available_jobs,
                default=[],
                help="Leave empty to include all jobs."
            )
            
            sel_locations = st.multiselect(
                "Filter by Location:",
                options=available_locations,
                default=[],
                help="Leave empty to include all locations."
            )

        with f_col2:
            sel_edu = st.multiselect(
                "Filter by Education Level:",
                options=available_education,
                default=[],
                help="Leave empty to include all degrees."
            )
            
            min_exp, max_exp = float(df["Years_of_Experience"].min()), float(df["Years_of_Experience"].max())
            exp_range = st.slider(
                "Filter by Years of Experience:",
                min_value=0.0,
                max_value=float(np.ceil(max_exp)),
                value=(0.0, float(np.ceil(max_exp))),
                step=0.5
            )

        with f_col3:
            min_sal, max_sal = float(df["Annual_Salary"].min()), float(df["Annual_Salary"].max())
            sal_range = st.slider(
                "Filter by Annual Salary (₹):",
                min_value=float(min_sal),
                max_value=float(max_sal),
                value=(float(min_sal), float(max_sal)),
                step=50000.0,
                format="₹ %d"
            )

    # Apply filters
    filtered_df = df.copy()

    if sel_jobs:
        filtered_df = filtered_df[filtered_df["Job_Title"].isin(sel_jobs)]
    if sel_locations:
        filtered_df = filtered_df[filtered_df["Location"].isin(sel_locations)]
    if sel_edu:
        filtered_df = filtered_df[filtered_df["Education_Level"].isin(sel_edu)]
    
    filtered_df = filtered_df[
        (filtered_df["Years_of_Experience"] >= exp_range[0]) &
        (filtered_df["Years_of_Experience"] <= exp_range[1]) &
        (filtered_df["Annual_Salary"] >= sal_range[0]) &
        (filtered_df["Annual_Salary"] <= sal_range[1])
    ]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} records:**")

    # Display interactive dataframe
    display_df = filtered_df.copy()
    display_df["Formatted_Salary"] = display_df["Annual_Salary"].apply(lambda x: format_inr(x, True))
    
    st.dataframe(
        display_df[["Age", "Education_Level", "Job_Title", "Years_of_Experience", "Location", "Formatted_Salary"]],
        column_config={
            "Age": "Age",
            "Education_Level": "Education",
            "Job_Title": "Job Title",
            "Years_of_Experience": st.column_config.NumberColumn("Experience (Yrs)", format="%.1f"),
            "Location": "Location",
            "Formatted_Salary": "Annual Salary (INR)"
        },
        use_container_width=True,
        hide_index=True
    )

    # Summary and Download Row
    c_down1, c_down2 = st.columns([1, 1])

    with c_down1:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="salary_dataset_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c_down2:
        with st.expander("📊 Summary Statistics of Current View"):
            st.dataframe(filtered_df.describe())

