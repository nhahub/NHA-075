from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import numpy as np
import requests  # Required for communicating with api.py

# Load Data
try:
    df = pd.read_csv("cleaned_data.csv")
except FileNotFoundError:
    # Fallback for demonstration if file is missing
    df = pd.DataFrame({
        'cardio': [0, 1], 'age_years': [50, 60], 'ap_hi': [120, 140], 'ap_lo': [80, 90],
        'cholesterol': [1, 2], 'glucose': [1, 2], 'bmi': [25, 30], 'pulse_pressure': [40, 50],
        'smoke': [0, 0], 'physically_active': [1, 0], 'weight': [70, 80], 'height': [170, 165]
    })
    print("Warning: 'cleaned_data.csv' not found. Using dummy data.")

# Calculate CVD prevalence percentage
cvd_percentage = (df['cardio'].value_counts(normalize=True) * 100).get(1, 0)


# 1. Gauge Chart
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
        'steps': [
            {'range': [0, 50], 'color': 'lightgreen'},
            {'range': [50, 100], 'color': 'salmon'}
        ],
    }
))
gauge_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    font={'color': "darkblue", 'family': "Arial"},
    height=350,
    margin=dict(l=20, r=20, t=60, b=20)
)

# 2. CVD Distribution
cvd_counts = df['cardio'].value_counts().sort_index()
cvd_fig = px.bar(
    x=['No CVD', 'Has CVD'],
    y=cvd_counts.values if not cvd_counts.empty else [0,0],
    color=['No CVD', 'Has CVD'],
    color_discrete_map={'No CVD': 'skyblue', 'Has CVD': 'salmon'},
    title="CVD Distribution",
    labels={'x': 'CVD Status', 'y': 'Count'}
)
cvd_fig.update_layout(
    showlegend=False, 
    height=350,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=60, b=20)
)

# 3. Age Distribution
if 'age_years' in df.columns:
    age_counts = df['age_years'].value_counts().sort_index()
    age_fig = px.bar(
        x=age_counts.index,
        y=age_counts.values,
        title="Age Distribution",
        labels={'x': 'Age (years)', 'y': 'Count'},
        color_discrete_sequence=['#3498db']
    )
    age_fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20)
    )
else:
    age_fig = go.Figure()

# 4. Correlation Heatmap
features = ['age_years', 'ap_hi', 'ap_lo', 'cholesterol', 'glucose', 'bmi',
            'pulse_pressure', 'smoke', 'physically_active', 'weight', 'cardio']
features = [f for f in features if f in df.columns]
correlation_matrix = df[features].corr()

heatmap_fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=features,
    y=features,
    text=correlation_matrix.round(2).astype(str).values,
    texttemplate='%{text}',
    textfont={"size": 10},
    colorscale='RdBu',
    zmid=0,
    colorbar=dict(title=dict(text='Correlation')), 
    hovertemplate='%{x} vs %{y}<br>corr=%{z:.2f}<extra></extra>'
))
heatmap_fig.update_layout(
    title='Feature Correlation Heatmap',
    title_x=0.5,
    height=700,
    xaxis_tickangle=-45,
    yaxis=dict(automargin=True),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='white',
    margin=dict(l=100, r=20, t=80, b=100)
)

# 5. Feature Importance
importance_features = ['ap_hi', 'age_years', 'glucose', 'cholesterol', 'bmi',
                       'pulse_pressure', 'smoke', 'physically_active', 'weight']
importance_features = [f for f in importance_features if f in df.columns]
if 'cardio' in df.columns:
    corr_with_cardio = df[importance_features].corrwith(df['cardio'])
    abs_corr = corr_with_cardio.abs().sort_values(ascending=True)
    colors_importance = ['red' if corr_with_cardio.loc[idx] > 0 else 'blue' for idx in abs_corr.index]

    importance_fig = go.Figure(go.Bar(
        x=abs_corr.values,
        y=abs_corr.index,
        orientation='h',
        marker=dict(color=colors_importance),
        hovertemplate='%{y}<br>corr=%{customdata:.3f}<extra></extra>',
        customdata=corr_with_cardio.loc[abs_corr.index].values
    ))
    importance_fig.update_layout(
        title='Feature Importance (|Pearson correlation with CVD|)',
        xaxis_title='Absolute correlation coefficient',
        yaxis_title='Feature',
        height=500,
        margin=dict(l=150, r=40, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white'
    )
else:
    importance_fig = go.Figure()

# 6. Scatter Matrix
scatter_cols = ['age_years', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
scatter_cols = [c for c in scatter_cols if c in df.columns]
if 'cardio' in df.columns:
    plot_df = df[scatter_cols + ['cardio']].copy()
    plot_df['cardio_label'] = plot_df['cardio'].map({0: 'Healthy', 1: 'CVD'})

    scatter_matrix_fig = px.scatter_matrix(
        plot_df,
        dimensions=scatter_cols,
        color='cardio_label',
        title='Scatter Plot Matrix — Pairwise Relationships',
        color_discrete_map={'Healthy': 'lightgreen', 'CVD': 'salmon'},
        labels={c: c.replace('_', ' ').title() for c in scatter_cols}
    )
    scatter_matrix_fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
    scatter_matrix_fig.update_layout(
        height=700, 
        legend_title_text='Outcome',
        paper_bgcolor='rgba(0,0,0,0)'
    )
else:
    scatter_matrix_fig = go.Figure()

# 7. Blood Pressure Categories
bp_fig = go.Figure()
if {'ap_hi', 'ap_lo', 'cardio'}.issubset(df.columns):
    def bp_category(row):
        s, d = row['ap_hi'], row['ap_lo']
        if s < 120 and d < 80: return 'Normal'
        if 120 <= s < 130 and d < 80: return 'Elevated'
        if (130 <= s < 140) or (80 <= d < 90): return 'Stage 1 HTN'
        return 'Stage 2 HTN'

    df['bp_category'] = df.apply(bp_category, axis=1)
    cats = ['Normal', 'Elevated', 'Stage 1 HTN', 'Stage 2 HTN']
    df['bp_category'] = pd.Categorical(df['bp_category'], categories=cats, ordered=True)

    bp_counts = df.groupby('bp_category')['cardio'].value_counts().unstack(fill_value=0).reindex(cats).fillna(0).astype(int)
    bp_counts = bp_counts.rename(columns={0: 'Healthy', 1: 'CVD'})
    bp_totals = bp_counts.sum(axis=1)
    bp_disease_pct = (bp_counts['CVD'] / bp_totals * 100).round(1).fillna(0)

    bp_fig.add_trace(go.Bar(y=cats, x=bp_counts['Healthy'], name='Healthy', orientation='h', marker_color='lightgreen'))
    bp_fig.add_trace(go.Bar(y=cats, x=bp_counts['CVD'], name='CVD', orientation='h', marker_color='salmon'))
    
    bp_annotations = []
    for i, cat in enumerate(cats):
        bp_annotations.append(dict(
            x=bp_totals.loc[cat] + max(bp_totals.max()*0.02, 1),  
            y=cat, text=f"{bp_disease_pct.loc[cat]}% CVD", showarrow=False,
            font=dict(size=11, color='black'), xanchor='left', yanchor='middle'
        ))
    bp_fig.update_layout(
        barmode='stack', title='Blood Pressure Categories and CVD Burden',
        xaxis_title='Number of Patients', yaxis_title='BP Category',
        height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        annotations=bp_annotations
    )

# 8. BMI Categories
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

    df['bmi_category'] = df['bmi'].apply(get_bmi_category)
    bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['bmi_category'] = pd.Categorical(df['bmi_category'], categories=bmi_categories)

    bmi_grouped = df.groupby('bmi_category')['cardio'].value_counts().unstack()
    if 1 in bmi_grouped.columns:
        bmi_totals = bmi_grouped.sum(axis=1)
        bmi_disease_pct = (bmi_grouped[1] / bmi_totals * 100).round(1)

        bmi_fig.add_trace(go.Bar(y=bmi_categories, x=bmi_grouped[0], name='Healthy', orientation='h', marker_color='lightgreen'))
        bmi_fig.add_trace(go.Bar(y=bmi_categories, x=bmi_grouped[1], name='CVD', orientation='h', marker_color='salmon'))

        bmi_annotations = []
        for i, cat in enumerate(bmi_categories):
            bmi_annotations.append(dict(
                x=bmi_totals.loc[cat] + max(bmi_totals.max()*0.02, 1),
                y=cat, text=f"{bmi_disease_pct.loc[cat]}% CVD", showarrow=False,
                font=dict(size=11, color='black'), xanchor='left', yanchor='middle'
            ))
        
        bmi_fig.update_layout(
            barmode='stack', title='BMI Categories and CVD Risk',
            xaxis_title='Number of Patients', yaxis_title='BMI Category',
            height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            annotations=bmi_annotations
        )

# 9. Metabolic Risk
metabolic_fig = go.Figure()
if {'cholesterol', 'glucose', 'cardio'}.issubset(df.columns):
    df['metabolic_score'] = df['cholesterol'] + df['glucose']
    healthy_scores = df[df['cardio'] == 0]['metabolic_score']
    cvd_scores = df[df['cardio'] == 1]['metabolic_score']

    metabolic_fig.add_trace(go.Histogram(x=healthy_scores, name='Healthy', nbinsx=30, opacity=0.7, marker_color='lightgreen', histnorm='probability'))
    metabolic_fig.add_trace(go.Histogram(x=cvd_scores, name='CVD', nbinsx=30, opacity=0.7, marker_color='salmon', histnorm='probability'))
    
    metabolic_fig.update_layout(
        title='Metabolic Risk Score Distribution', xaxis_title='Metabolic Risk Score (Cholesterol + Glucose)',
        yaxis_title='Probability', barmode='overlay', height=500,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white'
    )

# 10. CVD Prevalence by Age Group
age_prevalence_fig = go.Figure()
if {'age_years', 'cardio'}.issubset(df.columns):
    df['age_group'] = pd.cut(df['age_years'], bins=[0, 40, 50, 60, 70, 100], labels=['30-40', '40-50', '50-60', '60-70', '70+'])
    age_stats = df.groupby('age_group')['cardio'].mean().reset_index()
    age_stats['prevalence'] = age_stats['cardio'] * 100

    age_prevalence_fig.add_trace(go.Scatter(
        x=age_stats['age_group'], y=age_stats['prevalence'],
        mode='lines+markers', name='CVD Prevalence',
        line=dict(color='darkblue', width=2), marker=dict(size=8)
    ))
    age_prevalence_fig.update_layout(
        title='CVD Prevalence by Age Group', xaxis_title='Age Group (years)', yaxis_title='Disease Prevalence (%)',
        height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white'
    )

# 11. Age vs Systolic BP
age_bp_fig = go.Figure()
if {'age_years', 'ap_hi', 'cardio'}.issubset(df.columns):
    age_bp_fig = px.scatter(df, x='age_years', y='ap_hi', color='cardio', 
                           color_discrete_map={0: 'lightgreen', 1: 'salmon'},
                           title='Age vs Systolic Blood Pressure', size_max=15, opacity=0.6)
    age_bp_fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white')

# 12. BP Box Plots
bp_box_fig = go.Figure()
if {'ap_hi', 'ap_lo', 'pulse_pressure', 'cardio'}.issubset(df.columns):
    bp_box_fig = make_subplots(rows=1, cols=3, subplot_titles=('Systolic BP', 'Diastolic BP', 'Pulse Pressure'))
    for status, color, label in [(0, '#27ae60', 'Healthy'), (1, '#e74c3c', 'CVD')]:
        subset = df[df['cardio'] == status]
        bp_box_fig.add_trace(go.Box(y=subset['ap_hi'], name=label, marker_color=color, showlegend=(status==0)), row=1, col=1)
        bp_box_fig.add_trace(go.Box(y=subset['ap_lo'], name=label, marker_color=color, showlegend=False), row=1, col=2)
        bp_box_fig.add_trace(go.Box(y=subset['pulse_pressure'], name=label, marker_color=color, showlegend=False), row=1, col=3)
    bp_box_fig.update_layout(title="Blood Pressure Distributions", height=500, paper_bgcolor='rgba(0,0,0,0)')


# ==========================================
# ========== APP LAYOUT ====================
# ==========================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Cardiovascular Health Dashboard", 
                   className="text-center text-primary mb-4 mt-4",
                   style={'fontWeight': 'bold'})
        ])
    ]),

    # --- NEW PREDICTION SECTION (COMMUNICATES WITH API.PY) ---
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Live Risk Prediction", className="text-white mb-0"), className="bg-primary"),
                dbc.CardBody([
                    html.P("Enter patient details to calculate CVD risk using the backend model.", className="text-muted mb-3"),
                    
                    # Input Row 1
                    dbc.Row([
                        dbc.Col([html.Label("Age (Years)"), dbc.Input(id="in-age", type="number", value=50)], width=3),
                        dbc.Col([html.Label("Gender"), dbc.Select(id="in-gender", options=[{"label":"Female", "value":1}, {"label":"Male", "value":2}], value=1)], width=3),
                        dbc.Col([html.Label("Height (cm)"), dbc.Input(id="in-height", type="number", value=165)], width=3),
                        dbc.Col([html.Label("Weight (kg)"), dbc.Input(id="in-weight", type="number", value=70)], width=3),
                    ], className="mb-3"),

                    # Input Row 2
                    dbc.Row([
                        dbc.Col([html.Label("Systolic BP"), dbc.Input(id="in-aphi", type="number", value=120)], width=3),
                        dbc.Col([html.Label("Diastolic BP"), dbc.Input(id="in-aplo", type="number", value=80)], width=3),
                        dbc.Col([html.Label("Cholesterol"), dbc.Select(id="in-chol", options=[{"label":"Normal","value":1},{"label":"Above Normal","value":2},{"label":"High","value":3}], value=1)], width=3),
                        dbc.Col([html.Label("Glucose"), dbc.Select(id="in-gluc", options=[{"label":"Normal","value":1},{"label":"Above Normal","value":2},{"label":"High","value":3}], value=1)], width=3),
                    ], className="mb-3"),

                    # Input Row 3
                    dbc.Row([
                        dbc.Col([html.Label("Smoke?"), dbc.RadioItems(id="in-smoke", options=[{"label":"No","value":0},{"label":"Yes","value":1}], value=0, inline=True)], width=3),
                        dbc.Col([html.Label("Alcohol?"), dbc.RadioItems(id="in-alco", options=[{"label":"No","value":0},{"label":"Yes","value":1}], value=0, inline=True)], width=3),
                        dbc.Col([html.Label("Active?"), dbc.RadioItems(id="in-active", options=[{"label":"No","value":0},{"label":"Yes","value":1}], value=1, inline=True)], width=3),
                    ], className="mb-3"),

                    # Action Button
                    html.Div([
                        dbc.Button("Predict Risk", id="btn-predict", color="primary", size="lg", n_clicks=0, className="w-100")
                    ], className="d-grid gap-2 col-6 mx-auto"),

                    # Result Output
                    html.Div(id="api-output", className="mt-4 text-center")
                ])
            ], className="shadow mb-4")
        ], width=12)
    ]),

    html.Hr(),

    # --- EXISTING DASHBOARD CONTENT BELOW ---
    
    # Top cards with statistics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Patients", className="card-title"),
                    html.H2(f"{len(df):,}", className="text-primary")
                ])
            ], className="shadow-sm mb-4")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("CVD Cases", className="card-title"),
                    html.H2(f"{df['cardio'].sum():,}", className="text-danger")
                ])
            ], className="shadow-sm mb-4")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Age", className="card-title"),
                    html.H2(f"{df['age_years'].mean():.1f} years", className="text-success")
                ])
            ], className="shadow-sm mb-4")
        ], width=4)
    ]),
    
    # Overview Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=gauge_fig)])
            ], className="shadow-sm mb-4")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=cvd_fig)])
            ], className="shadow-sm mb-4")
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=age_fig)])
            ], className="shadow-sm mb-4")
        ], width=12)
    ]),
    
    # Correlation Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Correlation Analysis", className="mb-0")),
                dbc.CardBody([dcc.Graph(figure=heatmap_fig)])
            ], className="shadow-sm mb-4")
        ], width=12)
    ]),
    
    # Feature Importance & Scatter Matrix
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Feature Importance", className="mb-0")),
                dbc.CardBody([dcc.Graph(figure=importance_fig)])
            ], className="shadow-sm mb-4")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Pairwise Relationships", className="mb-0")),
                dbc.CardBody([dcc.Graph(figure=scatter_matrix_fig)])
            ], className="shadow-sm mb-4")
        ], width=6)
    ]),
    
    # Risk Categories
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Blood Pressure Categories", className="mb-0")),
                dbc.CardBody([dcc.Graph(figure=bp_fig)])
            ], className="shadow-sm mb-4")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("BMI Categories", className="mb-0")),
                dbc.CardBody([dcc.Graph(figure=bmi_fig)])
            ], className="shadow-sm mb-4")
        ], width=6)
    ]),
    
    # Metabolic & Age Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Metabolic Risk Score", className="mb-0")),
                dbc.CardBody([dcc.Graph(figure=metabolic_fig)])
            ], className="shadow-sm mb-4")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("CVD Prevalence by Age", className="mb-0")),
                dbc.CardBody([dcc.Graph(figure=age_prevalence_fig)])
            ], className="shadow-sm mb-4")
        ], width=6)
    ]),
    
    # Advanced Visualizations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Age vs Systolic BP Analysis", className="mb-0")),
                dbc.CardBody([dcc.Graph(figure=age_bp_fig)])
            ], className="shadow-sm mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Blood Pressure Distributions", className="mb-0")),
                dbc.CardBody([dcc.Graph(figure=bp_box_fig)])
            ], className="shadow-sm mb-4")
        ], width=12)
    ]),
    
    # Interactive Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Interactive Analysis", className="mb-0")),
                dbc.CardBody([
                    html.Label("Select a variable to analyze:", 
                              className="fw-bold mb-3",
                              style={'fontSize': '16px'}),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Lower Blood Pressure (ap_lo)', 'value': 'ap_lo'},
                            {'label': 'Upper Blood Pressure (ap_hi)', 'value': 'ap_hi'},
                            {'label': 'Body Mass Index (BMI)', 'value': 'bmi'},
                            {'label': 'Pulse Pressure', 'value': 'pulse_pressure'},
                            {'label': 'Age', 'value': 'age_years'},
                            {'label': 'Weight', 'value': 'weight'},
                            {'label': 'Height', 'value': 'height'}
                        ],
                        value='age_years',
                        id='controls_and_radioItems',
                        clearable=False
                    ),
                    html.Hr(),
                    dcc.Graph(figure={}, id='controls_and_graph')
                ])
            ], className="shadow-sm mb-4")
        ], width=12)
    ]),
    
    # Data Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Patient Data", className="mb-0")),
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df.to_dict('records'),
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'fontFamily': 'Arial'
                        },
                        style_header={
                            'backgroundColor': '#3498db',
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#f8f9fa'
                            }
                        ]
                    )
                ])
            ], className="shadow-sm mb-4")
        ], width=12)
    ])
    
], fluid=True, style={'backgroundColor': '#f5f7fa', 'minHeight': '100vh', 'paddingBottom': '50px'})

# --- CALLBACKS ---

# 1. API Prediction Callback
@callback(
    Output("api-output", "children"),
    Input("btn-predict", "n_clicks"),
    State("in-age", "value"), State("in-gender", "value"),
    State("in-height", "value"), State("in-weight", "value"),
    State("in-aphi", "value"), State("in-aplo", "value"),
    State("in-chol", "value"), State("in-gluc", "value"),
    State("in-smoke", "value"), State("in-alco", "value"),
    State("in-active", "value"),
    prevent_initial_call=True
)
def predict_with_api(n, age, gen, h, w, hi, lo, chol, gluc, smk, alc, act):
    # Payload matching FastAPI PatientData model
    payload = {
        "age_years": int(age), "gender": int(gen), "height": float(h),
        "weight": float(w), "ap_hi": int(hi), "ap_lo": int(lo),
        "cholesterol": int(chol), "glucose": int(gluc), "smoke": int(smk),
        "alcohol": int(alc), "physically_active": int(act)
    }

    try:
        # Send request to the backend API
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            risk = result["risk_probability"] * 100
            status = result["status"]
            
            color = "danger" if status == "High Risk" else "success"
            
            return html.Div([
                html.H3(f"{status} ({risk:.1f}%)", className=f"text-{color} fw-bold"),
                html.P("Prediction retrieved successfully from Backend API.", className="text-muted")
            ])
        else:
            return dbc.Alert(f"API Error: {response.text}", color="danger")
            
    except requests.exceptions.ConnectionError:
        return dbc.Alert("Connection failed! Ensure 'api.py' is running on port 8000.", color="warning")
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

# 2. Existing Graph Callback
@callback(
    Output(component_id='controls_and_graph', component_property='figure'),
    Input(component_id='controls_and_radioItems', component_property='value')
)
def update_graph(col_chosen):
    if col_chosen not in df.columns:
        return go.Figure()
        
    fig = px.histogram(
        df, 
        x=col_chosen, 
        color='cardio',
        title=f"Distribution of {col_chosen.replace('_', ' ').title()} by CVD Status",
        labels={'cardio': 'CVD Status', col_chosen: col_chosen.replace('_', ' ').title()},
        color_discrete_map={0: '#3498db', 1: '#e74c3c'},
        barmode='overlay'
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)