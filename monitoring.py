import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
import os

# Define file paths
REFERENCE_DATA_PATH = 'cleaned_data.csv'
CURRENT_DATA_PATH = 'prediction_log.csv'
REPORT_PATH = 'monitoring_report.html'

def generate_monitoring_report():

    print("Generating monitoring report...")

    # 1. Load Data
    if not os.path.exists(REFERENCE_DATA_PATH):
        print(f"Error: Reference data '{REFERENCE_DATA_PATH}' not found.")
        return
    
    if not os.path.exists(CURRENT_DATA_PATH):
        print(f"Error: Current data '{CURRENT_DATA_PATH}' not found. "
              "Make some predictions via the API first.")
        return

    try:
        reference_df = pd.read_csv(REFERENCE_DATA_PATH)
        current_df = pd.read_csv(CURRENT_DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Prepare Data for Evidently
    # Ensure column names match where applicable
    # 'cardio' is the target in the reference set
    # 'prediction_class' is the model's output in the current set
    
    # Rename for consistency in the report
    reference_df.rename(columns={'cardio': 'target'}, inplace=True)
    current_df.rename(columns={'prediction_class': 'prediction'}, inplace=True)

    # In a real-world scenario, the current data would eventually have a 'target' column
    # once ground truth is collected. For this simulation, we'll proceed without it
    # for drift analysis, but performance metrics will be limited.
    
    # Select common columns for drift analysis, excluding the target/prediction
    # as their concepts are different.
    profile_columns = [col for col in reference_df.columns if col in current_df.columns and col not in ['target', 'prediction']]
    
    if not profile_columns:
        print("Error: No common columns found for drift analysis.")
        return
    
    print(f"Found {len(profile_columns)} common features for analysis")

    # 3. Create Evidently Report
    print("Creating report...")
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset() # This will show prediction distribution and can show performance if 'target' is available
    ])
    
    # Run the report
    # Remove column_mapping - Evidently auto-detects 'target' and 'prediction' columns
    report.run(
        reference_data=reference_df[profile_columns + ['target']],
        current_data=current_df[profile_columns + ['prediction']]
    )

    # 4. Save the Report
    report.save_html(REPORT_PATH)
    print(f"Report saved to '{REPORT_PATH}'.")
    print(f"\nOpen '{REPORT_PATH}' in your browser to view the detailed analysis.")

if __name__ == '__main__':
    generate_monitoring_report()