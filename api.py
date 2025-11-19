from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import uvicorn

app = FastAPI(title="Cardio Prediction API")

MODEL_PATH = "artifacts/cvd_pipeline.joblib"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print("WARNING: Model not found. Run train_model.py first.")

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
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Convert input to DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    #Feature Engineering (Must match training exactly)
    # BMI
    height_m = df['height'] / 100
    df['bmi'] = df['weight'] / (height_m ** 2)
    
    # Pulse Pressure
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # Health Index
    df['health_index'] = df['physically_active'] - (0.5 * df['smoke']) - (0.5 * df['alcohol'])
    
    # Interaction
    df['cholesterol_gluc_interaction'] = df['cholesterol'] * df['glucose']
    
    # Hypertension
    df['hypertension'] = ((df['ap_hi'] >= 130) | (df['ap_lo'] >= 90)).astype(int)

    # Ensure column order matches training
    features = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
                'glucose', 'smoke', 'alcohol', 'physically_active',
                'age_years', 'bmi', 'pulse_pressure', 'health_index',
                'cholesterol_gluc_interaction', 'hypertension']
    
    df_final = df[features]

    #Predict
    try:
        probability = model.predict_proba(df_final)[0][1]
        prediction = int(model.predict(df_final)[0])
        
        return {
            "risk_probability": float(probability),
            "prediction_class": prediction,
            "status": "High Risk" if prediction == 1 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    # Run on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)