import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cardio Health Dashboard",
    page_icon="❤️",
    layout="wide"
)

# --- 1. LOAD RESOURCES (Model, Scaler, Data) ---
@st.cache_resource
def load_artifacts():
    # Load Model
    model_path = "artifacts/cvd_pipeline.joblib"
    scaler_path = "artifacts/scaler.joblib" # Make sure this file exists!
    
    model = None
    scaler = None
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        
    return model, scaler

@st.cache_data
def load_data():
    try:
        return pd.read_csv("cleaned_data.csv")
    except FileNotFoundError:
        return None

model, scaler = load_artifacts()
df = load_data()

# --- 2. SIDEBAR / INPUT FORM (Replaces the API Logic) ---
st.sidebar.header("Patient Input")
st.sidebar.write("Enter details for live prediction:")

with st.sidebar.form("prediction_form"):
    age = st.number_input("Age (Years)", 30, 100, 50)
    gender = st.selectbox("Gender", options=[("Female", 1), ("Male", 2)], format_func=lambda x: x[0])
    height = st.number_input("Height (cm)", 100, 250, 165)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    ap_hi = st.number_input("Systolic BP (ap_hi)", 60, 250, 120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", 40, 150, 80)
    cholesterol = st.selectbox("Cholesterol", options=[("Normal", 1), ("Above Normal", 2), ("High", 3)], format_func=lambda x: x[0])
    glucose = st.selectbox("Glucose", options=[("Normal", 1), ("Above Normal", 2), ("High", 3)], format_func=lambda x: x[0])
    smoke = st.radio("Smoker?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], horizontal=True)
    alcohol = st.radio("Alcohol Intake?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], horizontal=True)
    active = st.radio("Physically Active?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], horizontal=True)
    
    submit_btn = st.form_submit_button("Predict Risk")

# --- 3. PREDICTION LOGIC (Triggered by button) ---
if submit_btn:
    if model is None or scaler is None:
        st.error("Error: Model or Scaler file not found in 'artifacts/' folder.")
    else:
        # Create DataFrame from inputs
        input_data = pd.DataFrame([{
            'age_years': age, 'gender': gender[1], 'height': height, 'weight': weight,
            'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol[1],
            'glucose': glucose[1], 'smoke': smoke[1], 'alcohol': alcohol[1],
            'physically_active': active[1]
        }])

        # --- FEATURE ENGINEERING (Matches Notebook/API) ---
        input_data['bmi'] = input_data['weight'] / ((input_data['height'] / 100) ** 2)
        input_data['pulse_pressure'] = input_data['ap_hi'] - input_data['ap_lo']
        input_data['health_index'] = input_data['physically_active'] - (0.5 * input_data['smoke']) - (0.5 * input_data['alcohol'])
        input_data['cholesterol_gluc_interaction'] = input_data['cholesterol'] * input_data['glucose']
        input_data['hypertension'] = ((input_data['ap_hi'] >= 130) | (input_data['ap_lo'] >= 90)).astype(int)

        # Select columns in correct order
        features = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
                    'glucose', 'smoke', 'alcohol', 'physically_active',
                    'age_years', 'bmi', 'pulse_pressure', 'health_index',
                    'cholesterol_gluc_interaction', 'hypertension']
        
        df_final = input_data[features].copy()

        # --- SCALING ---
        continuous_features = ['height','weight','ap_hi','ap_lo','age_years','bmi','pulse_pressure']
        df_final[continuous_features] = scaler.transform(df_final[continuous_features])

        # Predict
        prob = model.predict_proba(df_final)[0][1] * 100
        pred_class = int(model.predict(df_final)[0])

        # Display Result
        st.divider()
        st.subheader("Prediction Result")
        if pred_class == 1:
            st.error(f"⚠️ HIGH RISK DETECTED (Probability: {prob:.1f}%)")
        else:
            st.success(f"✅ LOW RISK (Probability: {prob:.1f}%)")
        st.divider()


# --- 4. DASHBOARD SECTION ---
st.title("Cardiovascular Health Dashboard")

if df is None:
    st.warning("cleaned_data.csv not found. Showing demo mode.")
else:
    # Use Tabs for cleaner mobile view
    tab1, tab2, tab3 = st.tabs(["Overview", "Risk Factors", "Data Explorer"])

    with tab1:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", f"{len(df):,}")
        col2.metric("CVD Cases", f"{df['cardio'].sum():,}")
        col3.metric("Avg Age", f"{df['age_years'].mean():.1f} years")

        # Gauge & CVD Dist
        c1, c2 = st.columns(2)
        
        # Gauge
        cvd_percentage = (df['cardio'].value_counts(normalize=True) * 100).get(1, 0)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=cvd_percentage,
            title={'text': "CVD Prevalence (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': 'lightgreen'}, {'range': [50, 100], 'color': 'salmon'}]}
        ))
        c1.plotly_chart(fig_gauge, use_container_width=True)

        # Bar Chart
        cvd_counts = df['cardio'].value_counts().sort_index()
        fig_cvd = px.bar(x=['No CVD', 'Has CVD'], y=cvd_counts.values, 
                         color=['No CVD', 'Has CVD'], 
                         color_discrete_map={'No CVD': 'skyblue', 'Has CVD': 'salmon'},
                         title="CVD Distribution")
        c2.plotly_chart(fig_cvd, use_container_width=True)

    with tab2:
        st.subheader("Feature Analysis")
        
        # Feature Importance
        # (Simplified for Streamlit: Correlation)
        if 'cardio' in df.columns:
            features_corr = ['age_years', 'ap_hi', 'ap_lo', 'bmi', 'cholesterol', 'glucose']
            corr = df[features_corr].corrwith(df['cardio']).abs().sort_values()
            fig_imp = px.bar(x=corr.values, y=corr.index, orientation='h', title="Feature Correlation with CVD")
            st.plotly_chart(fig_imp, use_container_width=True)

        # BMI Categories
        if 'bmi' in df.columns:
            df['bmi_cat'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            fig_bmi = px.histogram(df, x='bmi_cat', color='cardio', barmode='group', 
                                   title="BMI Categories & Risk", color_discrete_map={0:'lightgreen', 1:'salmon'})
            st.plotly_chart(fig_bmi, use_container_width=True)

    with tab3:
        st.subheader("Interactive Explorer")
        x_axis = st.selectbox("Choose X-axis", options=['age_years', 'weight', 'ap_hi', 'bmi'])
        
        fig_hist = px.histogram(df, x=x_axis, color='cardio', barmode='overlay',
                                title=f"Distribution of {x_axis}",
                                color_discrete_map={0: '#3498db', 1: '#e74c3c'})
        st.plotly_chart(fig_hist, use_container_width=True)
        
        with st.expander("View Raw Data"):
            st.dataframe(df.head(100))