# ============================================
# AI-Powered Employee Attrition Prediction
# & Workforce Risk Intelligence System
# ============================================

# ============================================
# Import Required Libraries
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Employee Attrition Intelligence System",
    page_icon="📊",
    layout="wide"
)

# ============================================
# Dashboard Title
# ============================================

st.title(
    "AI-Powered Employee Attrition Prediction & Workforce Risk Intelligence System"
)

st.markdown("""
This dashboard provides predictive workforce analytics, employee attrition risk scoring,
and explainable AI insights to support proactive employee retention strategies.
""")

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("data/final_employee_attrition.csv")

# ============================================
# Load Trained Model
# ============================================

model = joblib.load("models/best_model.pkl")

# ============================================
# Initialize SHAP Explainer
# ============================================

explainer = shap.TreeExplainer(model)

# ============================================
# Sidebar Filters
# ============================================

st.sidebar.header("Dashboard Filters")

# Department Filter
selected_department = st.sidebar.selectbox(
    "Select Department",
    options=["All"] + list(df['Department'].unique())
)

# Apply Department Filter
if selected_department != "All":
    filtered_df = df[df['Department'] == selected_department]
else:
    filtered_df = df.copy()

# ============================================
# Prepare Feature Dataset
# ============================================

X = df.drop("Attrition", axis=1)

# ============================================
# Executive Workforce Overview
# ============================================

st.markdown("## Executive Workforce Overview")

st.markdown(
    "Real-time workforce intelligence and employee attrition monitoring"
)

# Calculate KPI values
total_employees = len(filtered_df)

attrition_cases = len(
    filtered_df[filtered_df['Attrition'] == 1]
)

attrition_rate = round(
    (attrition_cases / total_employees) * 100,
    2
)

average_income = round(
    filtered_df['MonthlyIncome'].mean(),
    2
)

# Create KPI columns
col1, col2, col3, col4 = st.columns(4)

# KPI Card 1
col1.metric(
    "Total Employees",
    total_employees
)

# KPI Card 2
col2.metric(
    "Attrition Cases",
    attrition_cases
)

# KPI Card 3
col3.metric(
    "Attrition Rate (%)",
    attrition_rate
)

# KPI Card 4
col4.metric(
    "Average Monthly Income",
    average_income
)

# ============================================
# Attrition Distribution Analysis
# ============================================

st.header("Attrition Distribution Analysis")

attrition_chart = px.histogram(
    filtered_df,
    x="Attrition",
    title="Employee Attrition Distribution"
)

st.plotly_chart(
    attrition_chart,
    use_container_width=True
)

# ============================================
# Department-wise Attrition Analysis
# ============================================

st.header("Department-wise Attrition Analysis")

department_chart = px.histogram(
    filtered_df,
    x="Department",
    color="Attrition",
    barmode="group",
    title="Department Attrition Comparison"
)

st.plotly_chart(
    department_chart,
    use_container_width=True
)

# ============================================
# Overtime Impact Analysis
# ============================================

st.header("Overtime Impact on Attrition")

overtime_chart = px.histogram(
    filtered_df,
    x="OverTime",
    color="Attrition",
    barmode="group",
    title="Overtime vs Employee Attrition"
)

st.plotly_chart(
    overtime_chart,
    use_container_width=True
)

# ============================================
# Employee Risk Prediction
# ============================================

st.header("Employee Risk Prediction")

# Employee selector
employee_index = st.selectbox(
    "Select Employee Index",
    filtered_df.index
)

# Extract employee data
employee_data = X.iloc[[employee_index]]

# Generate attrition probability
risk_probability = model.predict_proba(
    employee_data
)[0][1]

# Convert to percentage
risk_score = round(risk_probability * 100, 2)

# Determine risk category
if risk_score < 30:
    risk_category = "Low Risk"

elif risk_score < 60:
    risk_category = "Medium Risk"

else:
    risk_category = "High Risk"

# ============================================
# Display Risk Assessment
# ============================================

st.subheader("Employee Risk Assessment")

st.write(f"Risk Score: {risk_score}%")

st.write(f"Risk Category: {risk_category}")

# ============================================
# Risk Category Visualization
# ============================================

if risk_category == "Low Risk":
    st.success(f"Employee classified as {risk_category}")

elif risk_category == "Medium Risk":
    st.warning(f"Employee classified as {risk_category}")

else:
    st.error(f"Employee classified as {risk_category}")

# ============================================
# Risk Gauge Visualization
# ============================================

gauge_chart = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    title={'text': "Attrition Risk Score"},
    gauge={
        'axis': {'range': [0, 100]}
    }
))

st.plotly_chart(
    gauge_chart,
    use_container_width=True
)

# ============================================
# Explainable AI Insights
# ============================================

st.header("Explainable AI Insights")

st.markdown("""
This section explains the key factors influencing employee attrition predictions using SHAP (SHapley Additive exPlanations).
""")

# Generate SHAP values for selected employee
shap_values = explainer.shap_values(employee_data)

# ============================================
# Employee-Level SHAP Explanation
# ============================================

st.subheader("Employee-Level Prediction Explanation")

fig, ax = plt.subplots(figsize=(10, 5))

shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_values[0],
    employee_data.iloc[0],
    show=False
)

st.pyplot(fig)

# ============================================
# Global Feature Importance
# ============================================

st.subheader("Global Feature Importance")

# Generate SHAP values for complete dataset
global_shap_values = explainer.shap_values(X)

fig2, ax2 = plt.subplots(figsize=(10, 6))

shap.summary_plot(
    global_shap_values,
    X,
    plot_type="bar",
    show=False
)

st.pyplot(fig2)

# ============================================
# High-Risk Employee Analysis
# ============================================

st.header("High-Risk Employee Analysis")

# Generate probabilities for all employees
risk_probabilities = model.predict_proba(X)[:, 1]

# Create risk dataframe
risk_df = df.copy()

# Add probability percentage
risk_df['RiskProbability'] = (
    risk_probabilities * 100
)

# Create risk categories
risk_df['RiskCategory'] = np.where(
    risk_df['RiskProbability'] < 30,
    'Low Risk',
    np.where(
        risk_df['RiskProbability'] < 60,
        'Medium Risk',
        'High Risk'
    )
)

# Sort by highest risk
high_risk_table = risk_df.sort_values(
    by='RiskProbability',
    ascending=False
)

# Display top high-risk employees
st.dataframe(
    high_risk_table[[
        'RiskProbability',
        'RiskCategory',
        'MonthlyIncome',
        'JobSatisfaction',
        'WorkLifeBalance'
    ]].head(10)
)

# ============================================
# Download Workforce Risk Report
# ============================================

csv = high_risk_table.to_csv(index=False)

st.download_button(
    label="Download Workforce Risk Report",
    data=csv,
    file_name='employee_attrition_risk_report.csv',
    mime='text/csv'
)

# ============================================
# AI-Driven Workforce Recommendations
# ============================================

st.header("AI-Driven Workforce Recommendations")

st.markdown("""
### Key Workforce Insights

- Employees with excessive overtime demonstrate higher attrition tendencies.
- Lower job satisfaction strongly contributes to employee turnover risk.
- Promotion delays may negatively impact employee retention.
- Work-life imbalance is associated with elevated attrition probability.

### Recommended Actions

- Improve employee wellness initiatives
- Reduce excessive overtime workloads
- Strengthen career development programs
- Enhance employee engagement strategies
- Implement proactive retention planning
""")

# ============================================
# Dashboard Footer
# ============================================

st.markdown("---")

st.markdown("""
### Workforce Intelligence Summary

This AI-powered platform enables proactive employee retention by combining:

- Predictive analytics
- Workforce risk scoring
- Explainable AI
- Interactive HR intelligence dashboards

Developed as part of the Unified Mentor Internship Program.
""")