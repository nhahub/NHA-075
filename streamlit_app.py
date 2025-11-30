import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import numpy as np
import os
import csv
import threading
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cardiovascular Health Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOCK FOR THREAD-SAFE LOGGING ---
log_lock = threading.Lock()
LOG_FILE = "prediction_log.csv"

# --- 1. LOAD RESOURCES (Model, Scaler, Data) ---
@st.cache_resource
def load_artifacts():
    # Load Model and Scaler
    model_path = "artifacts/cvd_pipeline.joblib"
    scaler_path = "artifacts/scaler.joblib" 
    
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
        df = pd.read_csv("cleaned_data.csv")
        return df
    except FileNotFoundError:
        # Dummy data fallback to prevent crash
        return pd.DataFrame({
            'cardio': [0, 1] * 50, 'age_years': np.random.randint(40, 70, 100), 
            'ap_hi': np.random.randint(110, 150, 100), 'ap_lo': np.random.randint(70, 100, 100),
            'cholesterol': np.random.choice([1, 2, 3], 100), 'glucose': np.random.choice([1, 2, 3], 100), 
            'bmi': np.random.uniform(20, 35, 100), 'pulse_pressure': np.random.randint(30, 60, 100),
            'smoke': np.random.choice([0, 1], 100), 'physically_active': np.random.choice([0, 1], 100), 
            'weight': np.random.uniform(60, 90, 100), 'height': np.random.uniform(160, 180, 100)
        })

model, scaler = load_artifacts()
df = load_data()

# --- CSS STYLING ---
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h1 {color: #0d6efd; text-align: center; font-weight: bold;}
    .stButton>button {width: 100%; background-color: #0d6efd; color: white; font-weight: bold;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Cardiovascular Health Dashboard")

# ==========================================
# PART 1: LIVE PREDICTION & LOGGING
# ==========================================
st.markdown("### 🏥 Live Risk Prediction")
with st.container(border=True):
    st.info("Enter patient details below to calculate CVD risk. Data will be logged for monitoring.")
    
    # Input Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1: age = st.number_input("Age (Years)", 30, 100, 50)
    with c2: gender = st.selectbox("Gender", options=[("Female", 1), ("Male", 2)], format_func=lambda x: x[0])
    with c3: height = st.number_input("Height (cm)", 100, 250, 165)
    with c4: weight = st.number_input("Weight (kg)", 30, 200, 70)

    # Input Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5: ap_hi = st.number_input("Systolic BP", 60, 250, 120)
    with c6: ap_lo = st.number_input("Diastolic BP", 40, 150, 80)
    with c7: cholesterol = st.selectbox("Cholesterol", options=[("Normal", 1), ("Above Normal", 2), ("High", 3)], format_func=lambda x: x[0])
    with c8: glucose = st.selectbox("Glucose", options=[("Normal", 1), ("Above Normal", 2), ("High", 3)], format_func=lambda x: x[0])

    # Input Row 3
    c9, c10, c11 = st.columns(3)
    with c9: smoke = st.radio("Smoke?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], horizontal=True)
    with c10: alcohol = st.radio("Alcohol?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], horizontal=True)
    with c11: active = st.radio("Active?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], horizontal=True)

    # Predict Button
    if st.button("Predict Risk"):
        if model is None or scaler is None:
            st.error("❌ Model or Scaler not found! Please check 'artifacts/' folder.")
        else:
            # 1. Create DataFrame
            input_dict = {
                'age_years': age, 'gender': gender[1], 'height': height, 'weight': weight,
                'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol[1],
                'glucose': glucose[1], 'smoke': smoke[1], 'alcohol': alcohol[1],
                'physically_active': active[1]
            }
            input_data = pd.DataFrame([input_dict])

            # 2. Feature Engineering (Must match Training/API exactly)
            input_data['bmi'] = input_data['weight'] / ((input_data['height'] / 100) ** 2)
            input_data['pulse_pressure'] = input_data['ap_hi'] - input_data['ap_lo']
            input_data['health_index'] = input_data['physically_active'] - (0.5 * input_data['smoke']) - (0.5 * input_data['alcohol'])
            input_data['cholesterol_gluc_interaction'] = input_data['cholesterol'] * input_data['glucose']
            input_data['hypertension'] = ((input_data['ap_hi'] >= 130) | (input_data['ap_lo'] >= 90)).astype(int)

            features = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
                        'glucose', 'smoke', 'alcohol', 'physically_active',
                        'age_years', 'bmi', 'pulse_pressure', 'health_index',
                        'cholesterol_gluc_interaction', 'hypertension']
            
            df_final = input_data[features].copy()

            # 3. Scaling
            continuous_features = ['height','weight','ap_hi','ap_lo','age_years','bmi','pulse_pressure']
            try:
                df_final[continuous_features] = scaler.transform(df_final[continuous_features])
                
                # 4. Prediction
                prob = model.predict_proba(df_final)[0][1]
                pred_class = int(model.predict(df_final)[0])
                prob_pct = prob * 100
                
                status = "High Risk" if pred_class == 1 else "Low Risk"
                color = "#e74c3c" if pred_class == 1 else "#2ecc71"
                
                # 5. Display Result
                st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                    <h2 style='margin:0; color: white;'>{status}</h2>
                    <h3 style='margin:0; color: white;'>Probability: {prob_pct:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 6. LOGGING (Replicating api.py functionality)
                try:
                    with log_lock:
                        log_entry = input_dict.copy()
                        log_entry['prediction_probability'] = prob
                        log_entry['prediction_class'] = pred_class
                        log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        file_exists = os.path.isfile(LOG_FILE)
                        with open(LOG_FILE, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                            if not file_exists:
                                writer.writeheader()
                            writer.writerow(log_entry)
                    st.success("✅ Prediction logged successfully.")
                except Exception as log_err:
                    st.warning(f"⚠️ Prediction shown, but logging failed: {log_err}")

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

st.divider()

# ==========================================
# PART 2: DASHBOARD STATISTICS & VISUALS
# ==========================================

# 1. Top Statistics Cards
col_stat1, col_stat2, col_stat3 = st.columns(3)
with col_stat1:
    st.metric("Total Patients", f"{len(df):,}")
with col_stat2:
    st.metric("CVD Cases", f"{df['cardio'].sum():,}", delta_color="inverse")
with col_stat3:
    st.metric("Avg Age", f"{df['age_years'].mean():.1f} years")

st.write("") # Spacer

# 2. Gauge & Distribution (Side by Side)
col_g1, col_g2 = st.columns(2)

with col_g1:
    with st.container(border=True):
        cvd_percentage = (df['cardio'].value_counts(normalize=True) * 100).get(1, 0)
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cvd_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CVD Prevalence (%)", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [{'range': [0, 50], 'color': 'lightgreen'}, {'range': [50, 100], 'color': 'salmon'}]
            }
        ))
        gauge_fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

with col_g2:
    with st.container(border=True):
        cvd_counts = df['cardio'].value_counts().sort_index()
        cvd_fig = px.bar(
            x=['No CVD', 'Has CVD'],
            y=cvd_counts.values if not cvd_counts.empty else [0,0],
            color=['No CVD', 'Has CVD'],
            color_discrete_map={'No CVD': 'skyblue', 'Has CVD': 'salmon'},
            title="CVD Distribution",
            labels={'x': 'CVD Status', 'y': 'Count'}
        )
        cvd_fig.update_layout(showlegend=False, height=350, margin=dict(t=50))
        st.plotly_chart(cvd_fig, use_container_width=True)

# 3. Age Distribution
with st.container(border=True):
    if 'age_years' in df.columns:
        age_counts = df['age_years'].value_counts().sort_index()
        age_fig = px.bar(
            x=age_counts.index, y=age_counts.values,
            title="Age Distribution",
            labels={'x': 'Age (years)', 'y': 'Count'},
            color_discrete_sequence=['#3498db']
        )
        age_fig.update_layout(height=350)
        st.plotly_chart(age_fig, use_container_width=True)

# 4. Correlation Heatmap
with st.container(border=True):
    st.subheader("Correlation Analysis")
    features = ['age_years', 'ap_hi', 'ap_lo', 'cholesterol', 'glucose', 'bmi',
                'pulse_pressure', 'smoke', 'physically_active', 'weight', 'cardio']
    features = [f for f in features if f in df.columns]
    correlation_matrix = df[features].corr()

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values, x=features, y=features,
        text=correlation_matrix.round(2).astype(str).values,
        texttemplate='%{text}', textfont={"size": 10},
        colorscale='RdBu', zmid=0,
        colorbar=dict(title=dict(text='Correlation')),
        hovertemplate='%{x} vs %{y}<br>corr=%{z:.2f}<extra></extra>'
    ))
    heatmap_fig.update_layout(height=700, xaxis_tickangle=-45, title_x=0.5)
    st.plotly_chart(heatmap_fig, use_container_width=True)

# 5. Feature Importance & Scatter Matrix
col_fi1, col_fi2 = st.columns(2)

with col_fi1:
    with st.container(border=True):
        st.subheader("Feature Importance")
        importance_features = ['ap_hi', 'age_years', 'glucose', 'cholesterol', 'bmi',
                               'pulse_pressure', 'smoke', 'physically_active', 'weight']
        importance_features = [f for f in importance_features if f in df.columns]
        
        if 'cardio' in df.columns:
            corr_with_cardio = df[importance_features].corrwith(df['cardio'])
            abs_corr = corr_with_cardio.abs().sort_values(ascending=True)
            colors_importance = ['red' if corr_with_cardio.loc[idx] > 0 else 'blue' for idx in abs_corr.index]

            importance_fig = go.Figure(go.Bar(
                x=abs_corr.values, y=abs_corr.index, orientation='h',
                marker=dict(color=colors_importance),
                customdata=corr_with_cardio.loc[abs_corr.index].values
            ))
            importance_fig.update_layout(
                title='Feature Importance (|Pearson correlation with CVD|)',
                xaxis_title='Absolute correlation coefficient', height=500
            )
            st.plotly_chart(importance_fig, use_container_width=True)

with col_fi2:
    with st.container(border=True):
        st.subheader("Pairwise Relationships")
        scatter_cols = ['age_years', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
        scatter_cols = [c for c in scatter_cols if c in df.columns]
        if 'cardio' in df.columns:
            # Downsample for Scatter Matrix speed if df is huge
            plot_df = df[scatter_cols + ['cardio']].sample(min(1000, len(df))).copy()
            plot_df['cardio_label'] = plot_df['cardio'].map({0: 'Healthy', 1: 'CVD'})

            scatter_matrix_fig = px.scatter_matrix(
                plot_df, dimensions=scatter_cols, color='cardio_label',
                title='Scatter Plot Matrix (Sampled)',
                color_discrete_map={'Healthy': 'lightgreen', 'CVD': 'salmon'},
                labels={c: c.replace('_', ' ').title() for c in scatter_cols}
            )
            scatter_matrix_fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
            scatter_matrix_fig.update_layout(height=700, legend_title_text='Outcome', dragmode='select')
            st.plotly_chart(scatter_matrix_fig, use_container_width=True)

# 6. Risk Categories (BP & BMI)
col_rc1, col_rc2 = st.columns(2)

with col_rc1:
    with st.container(border=True):
        bp_fig = go.Figure()
        if {'ap_hi', 'ap_lo', 'cardio'}.issubset(df.columns):
            def bp_category(row):
                s, d = row['ap_hi'], row['ap_lo']
                if s < 120 and d < 80: return 'Normal'
                if 120 <= s < 130 and d < 80: return 'Elevated'
                if (130 <= s < 140) or (80 <= d < 90): return 'Stage 1 HTN'
                return 'Stage 2 HTN'

            if 'bp_category' not in df.columns:
                df['bp_category'] = df.apply(bp_category, axis=1)
                
            cats = ['Normal', 'Elevated', 'Stage 1 HTN', 'Stage 2 HTN']
            df_bp_cat = df.copy()
            df_bp_cat['bp_category'] = pd.Categorical(df_bp_cat['bp_category'], categories=cats, ordered=True)
            bp_counts = df_bp_cat.groupby('bp_category', observed=False)['cardio'].value_counts().unstack(fill_value=0).reindex(cats).fillna(0)
            
            bp_fig.add_trace(go.Bar(y=cats, x=bp_counts[0], name='Healthy', orientation='h', marker_color='lightgreen'))
            bp_fig.add_trace(go.Bar(y=cats, x=bp_counts[1], name='CVD', orientation='h', marker_color='salmon'))
            bp_fig.update_layout(barmode='stack', title='Blood Pressure Categories', height=450, legend=dict(orientation='h', y=1.1))
            st.plotly_chart(bp_fig, use_container_width=True)

with col_rc2:
    with st.container(border=True):
        bmi_fig = go.Figure()
        if 'bmi' in df.columns:
            def get_bmi_category(bmi):
                try:
                    bmi_float = float(bmi)
                    if bmi_float < 18.5: return 'Underweight'
                    elif bmi_float < 25: return 'Normal'
                    elif bmi_float < 30: return 'Overweight'
                    else: return 'Obese'
                except: return 'Unknown'
            
            if 'bmi_category' not in df.columns:
                df['bmi_category'] = df['bmi'].apply(get_bmi_category)
            
            bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
            df_bmi_cat = df.copy()
            df_bmi_cat['bmi_category'] = pd.Categorical(df_bmi_cat['bmi_category'], categories=bmi_categories)
            bmi_grouped = df_bmi_cat.groupby('bmi_category', observed=False)['cardio'].value_counts().unstack().fillna(0)
            
            if 1 in bmi_grouped.columns:
                bmi_fig.add_trace(go.Bar(y=bmi_categories, x=bmi_grouped[0], name='Healthy', orientation='h', marker_color='lightgreen'))
                bmi_fig.add_trace(go.Bar(y=bmi_categories, x=bmi_grouped[1], name='CVD', orientation='h', marker_color='salmon'))
                bmi_fig.update_layout(barmode='stack', title='BMI Categories', height=450, legend=dict(orientation='h', y=1.1))
                st.plotly_chart(bmi_fig, use_container_width=True)

# 7. Metabolic & Age Analysis
col_ma1, col_ma2 = st.columns(2)

with col_ma1:
    with st.container(border=True):
        metabolic_fig = go.Figure()
        if {'cholesterol', 'glucose', 'cardio'}.issubset(df.columns):
            df['metabolic_score'] = df['cholesterol'] + df['glucose']
            healthy_scores = df[df['cardio'] == 0]['metabolic_score']
            cvd_scores = df[df['cardio'] == 1]['metabolic_score']

            metabolic_fig.add_trace(go.Histogram(x=healthy_scores, name='Healthy', nbinsx=30, opacity=0.7, marker_color='lightgreen', histnorm='probability'))
            metabolic_fig.add_trace(go.Histogram(x=cvd_scores, name='CVD', nbinsx=30, opacity=0.7, marker_color='salmon', histnorm='probability'))
            metabolic_fig.update_layout(title='Metabolic Risk Score Distribution', barmode='overlay', height=500)
            st.plotly_chart(metabolic_fig, use_container_width=True)

with col_ma2:
    with st.container(border=True):
        age_prevalence_fig = go.Figure()
        if {'age_years', 'cardio'}.issubset(df.columns):
            df['age_group'] = pd.cut(df['age_years'], bins=[0, 40, 50, 60, 70, 100], labels=['30-40', '40-50', '50-60', '60-70', '70+'])
            age_stats = df.groupby('age_group', observed=False)['cardio'].mean().reset_index()
            age_stats['prevalence'] = age_stats['cardio'] * 100

            age_prevalence_fig.add_trace(go.Scatter(
                x=age_stats['age_group'], y=age_stats['prevalence'],
                mode='lines+markers', name='CVD Prevalence',
                line=dict(color='darkblue', width=2), marker=dict(size=8)
            ))
            age_prevalence_fig.update_layout(title='CVD Prevalence by Age Group', xaxis_title='Age Group', yaxis_title='Disease Prevalence (%)', height=500)
            st.plotly_chart(age_prevalence_fig, use_container_width=True)

# 8. Advanced Visualizations (BP & Box Plots)
with st.container(border=True):
    age_bp_fig = go.Figure()
    if {'age_years', 'ap_hi', 'cardio'}.issubset(df.columns):
        plot_df_scatter = df.sample(min(2000, len(df)))
        age_bp_fig = px.scatter(plot_df_scatter, x='age_years', y='ap_hi', color='cardio', 
                                color_discrete_map={0: 'lightgreen', 1: 'salmon'},
                                title='Age vs Systolic Blood Pressure', size_max=15, opacity=0.6)
        age_bp_fig.update_layout(height=600)
        st.plotly_chart(age_bp_fig, use_container_width=True)

with st.container(border=True):
    bp_box_fig = go.Figure()
    if {'ap_hi', 'ap_lo', 'pulse_pressure', 'cardio'}.issubset(df.columns):
        bp_box_fig = make_subplots(rows=1, cols=3, subplot_titles=('Systolic BP', 'Diastolic BP', 'Pulse Pressure'))
        for status, color, label in [(0, '#27ae60', 'Healthy'), (1, '#e74c3c', 'CVD')]:
            subset = df[df['cardio'] == status]
            bp_box_fig.add_trace(go.Box(y=subset['ap_hi'], name=label, marker_color=color, showlegend=(status==0)), row=1, col=1)
            bp_box_fig.add_trace(go.Box(y=subset['ap_lo'], name=label, marker_color=color, showlegend=False), row=1, col=2)
            bp_box_fig.add_trace(go.Box(y=subset['pulse_pressure'], name=label, marker_color=color, showlegend=False), row=1, col=3)
        bp_box_fig.update_layout(title="Blood Pressure Distributions", height=500)
        st.plotly_chart(bp_box_fig, use_container_width=True)

# 9. Interactive Analysis
st.subheader("Interactive Analysis")
with st.container(border=True):
    col_chosen = st.selectbox(
        "Select a variable to analyze:",
        options=['ap_lo', 'ap_hi', 'bmi', 'pulse_pressure', 'age_years', 'weight', 'height'],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if col_chosen in df.columns:
        fig_inter = px.histogram(
            df, x=col_chosen, color='cardio',
            title=f"Distribution of {col_chosen.replace('_', ' ').title()} by CVD Status",
            labels={'cardio': 'CVD Status', col_chosen: col_chosen.replace('_', ' ').title()},
            color_discrete_map={0: '#3498db', 1: '#e74c3c'},
            barmode='overlay'
        )
        fig_inter.update_layout(height=400)
        st.plotly_chart(fig_inter, use_container_width=True)

# 10. Data Table
st.subheader("Patient Data Records")
with st.expander("View Raw Data", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)