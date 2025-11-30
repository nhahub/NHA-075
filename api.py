from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import uvicorn
import csv
import threading

app = FastAPI(title="Cardio Prediction API")

# 1. ADD SCALER PATH
MODEL_PATH = "artifacts/cvd_pipeline.joblib"
SCALER_PATH = "artifacts/cvd_scaler.joblib" 
LOG_FILE = "prediction_log.csv"

model = None
scaler = None 
log_lock = threading.Lock()

# Load Model AND Scaler
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print("WARNING: Model not found.")

if os.path.exists(SCALER_PATH):  
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
else:
    print("WARNING: Scaler not found. You must save the scaler from your training notebook.")

class PatientData(BaseModel):
    age_years: int
    gender: int
    height: float
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    glucose: int
    smoke: int
    alcohol: int
    physically_active: int

@app.get("/")
def home():
    return {"message": "Cardiovascular Risk Prediction API is running."}

@app.post("/predict")
def predict_risk(data: PatientData):
    if model is None or scaler is None: # <--- Check for scaler
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded.")

    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # --- Feature Engineering (Matches Training) ---
    height_m = df['height'] / 100
    df['bmi'] = df['weight'] / (height_m ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # Verify this formula matches your notebook exactly. 
    # In snippet 54 you used: df['lifestyle_risk'] = df['smoke'] + (1 - df['physically_active']) + df['alcohol']
    # In snippet 4 you used: health_index = physically_active - ...
    # USE THE ONE FROM YOUR NOTEBOOK THAT GENERATED 'health_index'
    df['health_index'] = df['physically_active'] - (0.5 * df['smoke']) - (0.5 * df['alcohol'])
    
    df['cholesterol_gluc_interaction'] = df['cholesterol'] * df['glucose']
    df['hypertension'] = ((df['ap_hi'] >= 130) | (df['ap_lo'] >= 90)).astype(int)

    # Define feature sets
    features = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
                'glucose', 'smoke', 'alcohol', 'physically_active',
                'age_years', 'bmi', 'pulse_pressure', 'health_index',
                'cholesterol_gluc_interaction', 'hypertension']
    
    # List of continuous features to scale (MUST match notebook snippet 80)
    continuous_features = ['height','weight','ap_hi','ap_lo','age_years','bmi','pulse_pressure']

    df_final = df[features].copy()

    # 2. APPLY SCALING
    # The scaler expects only the continuous columns, but it's safer to transform 
    # strictly the columns it was trained on.
    try:
        df_final[continuous_features] = scaler.transform(df_final[continuous_features])
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Scaling error: {str(e)}")

    # Predict
    try:
        # Ensure column order exactly matches training X.columns
        probability = model.predict_proba(df_final)[0][1]
        prediction = int(model.predict(df_final)[0])
        
        # Log the prediction
        with log_lock:
            log_data = input_dict.copy()
            log_data['prediction_probability'] = probability
            log_data['prediction_class'] = prediction
            
            file_exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)

        return {
            "risk_probability": float(probability),
            "prediction_class": prediction,
            "status": "High Risk" if prediction == 1 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)