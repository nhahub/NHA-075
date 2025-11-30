import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import os

REFERENCE_DATA_PATH = "cleaned_data.csv"
CURRENT_DATA_PATH = "prediction_log.csv"
REPORT_PATH = "monitoring_report.html"


def generate_monitoring_report():
    """
    Generate Evidently monitoring report comparing reference and current data
    Compatible with Evidently 0.7.x (new API)
    """
    print("=" * 60)
    print("Generating ML Monitoring Report")
    print("=" * 60)

    # ---- LOAD DATA ----
    if not os.path.exists(REFERENCE_DATA_PATH):
        print(f"❌ Reference data not found: {REFERENCE_DATA_PATH}")
        return False

    if not os.path.exists(CURRENT_DATA_PATH):
        print(f"❌ Current data not found: {CURRENT_DATA_PATH}")
        return False

    try:
        reference_df = pd.read_csv(REFERENCE_DATA_PATH)
        current_df = pd.read_csv(CURRENT_DATA_PATH)
        
        print(f"✓ Loaded reference data: {reference_df.shape}")
        print(f"✓ Loaded current data: {current_df.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

    # ---- ALIGN COLUMNS ----
    # Remove columns that exist in current but not in reference
    # (e.g., timestamp, prediction_id from logs)
    extra_cols = set(current_df.columns) - set(reference_df.columns)
    if extra_cols:
        print(f"\nℹ Removing extra columns from current data: {extra_cols}")
        current_df = current_df.drop(columns=list(extra_cols))
    
    # Check for missing columns
    missing_cols = set(reference_df.columns) - set(current_df.columns)
    if missing_cols:
        print(f"⚠ Warning: Missing columns in current data: {missing_cols}")
        # Fill missing columns with NaN
        for col in missing_cols:
            current_df[col] = pd.NA
    
    # Ensure same column order
    current_df = current_df[reference_df.columns]
    
    print(f"\n✓ Aligned data shapes:")
    print(f"  Reference: {reference_df.shape}")
    print(f"  Current: {current_df.shape}")

    # ---- ENSURE NUMERIC TYPES ----
    # Evidently 0.7.x automatically detects column types
    # Just make sure data types are consistent
    for col in reference_df.columns:
        if reference_df[col].dtype != current_df[col].dtype:
            try:
                current_df[col] = current_df[col].astype(reference_df[col].dtype)
            except:
                print(f"⚠ Warning: Could not convert {col} to matching type")

    print(f"\n✓ Data types aligned")

    # ---- CREATE REPORT ----
    # In Evidently 0.7.x, column mapping is automatic
    # Just pass the dataframes directly
    try:
        report = Report(metrics=[
            DataDriftPreset(),
            DataSummaryPreset()
        ])

        print("\n⏳ Running drift detection...")
        
        # ---- RUN REPORT ----
        report.run(
            reference_data=reference_df,
            current_data=current_df
        )

        # ---- SAVE ----
        report.save_html(REPORT_PATH)
        print(f"\n✅ Report successfully saved to: {REPORT_PATH}")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_monitoring_report()
    exit(0 if success else 1)