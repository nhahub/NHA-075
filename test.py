import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import joblib
import os
import json

# Load the fixed data
print("Loading cleaned_data_fixed.csv...")
df = pd.read_csv("cleaned_data_fixed.csv")
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Define features (matching your metadata)
features = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
            'glucose', 'smoke', 'alcohol', 'physically_active',
            'age_years', 'bmi', 'pulse_pressure', 'health_index',
            'cholesterol_gluc_interaction', 'hypertension']

target = 'cardio'

# Prepare data
X = df[features]
y = df[target]

print("\n" + "="*60)
print("DATA SUMMARY:")
print("="*60)
print(f"Features: {len(features)}")
print(f"Samples: {len(X)}")
print(f"CVD Rate: {y.mean():.2%}")

# Check for any missing values
print(f"\nMissing values per feature:")
for col in features:
    missing = X[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train Decision Tree (your best model)
print("\n" + "="*60)
print("TRAINING DECISION TREE MODEL...")
print("="*60)

model = DecisionTreeClassifier(random_state=42)
pipeline = Pipeline([
    ("model", model)
])

pipeline.fit(X_train, y_train)
print("✅ Training complete!")

# Evaluate
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*60)
print("MODEL PERFORMANCE:")
print("="*60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Test on sample cases
print("\n" + "="*60)
print("TESTING SAMPLE CASES:")
print("="*60)

test_cases = [
    {
        "name": "Low Risk Patient",
        "data": {
            'gender': 1, 'height': 165.0, 'weight': 70.0, 'ap_hi': 120, 'ap_lo': 80,
            'cholesterol': 1, 'glucose': 1, 'smoke': 0, 'alcohol': 0, 'physically_active': 1,
            'age_years': 50, 'bmi': 25.71, 'pulse_pressure': 40, 'health_index': 1.0,
            'cholesterol_gluc_interaction': 1, 'hypertension': 0
        }
    },
    {
        "name": "High Risk Patient",
        "data": {
            'gender': 2, 'height': 170.0, 'weight': 95.0, 'ap_hi': 160, 'ap_lo': 100,
            'cholesterol': 3, 'glucose': 3, 'smoke': 1, 'alcohol': 1, 'physically_active': 0,
            'age_years': 65, 'bmi': 32.87, 'pulse_pressure': 60, 'health_index': -1.0,
            'cholesterol_gluc_interaction': 9, 'hypertension': 1
        }
    },
    {
        "name": "Very Young, Healthy",
        "data": {
            'gender': 1, 'height': 168.0, 'weight': 65.0, 'ap_hi': 110, 'ap_lo': 70,
            'cholesterol': 1, 'glucose': 1, 'smoke': 0, 'alcohol': 0, 'physically_active': 1,
            'age_years': 35, 'bmi': 23.03, 'pulse_pressure': 40, 'health_index': 1.0,
            'cholesterol_gluc_interaction': 1, 'hypertension': 0
        }
    }
]

for case in test_cases:
    df_test = pd.DataFrame([case['data']])[features]
    prob = pipeline.predict_proba(df_test)[0][1]
    pred = pipeline.predict(df_test)[0]
    print(f"{case['name']:20s} | Risk: {prob:6.2%} | Class: {pred}")

# Save the model
print("\n" + "="*60)
print("SAVING MODEL...")
print("="*60)

os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/cvd_best_pipeline.joblib"
joblib.dump(pipeline, model_path)
print(f"✅ Model saved to: {model_path}")

# Save metadata
metadata = {
    "best_model": "Decision Tree",
    "recall": float(recall),
    "features": features,
    "selection_metric": "recall",
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "cvd_rate": float(y.mean()),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc)
}

metadata_path = "artifacts/metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved to: {metadata_path}")

print("\n" + "="*60)
print("✅ MODEL RETRAINING COMPLETE!")
print("="*60)
print("\nYou can now:")
print("1. Restart your API: python api.py")
print("2. Test predictions in your Dash app")
print("3. The model should now give reasonable predictions!")