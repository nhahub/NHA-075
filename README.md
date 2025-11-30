# NHA-075
Cardiovascular Disease Risk Prediction Dashboard

A comprehensive Machine Learning powered dashboard that predicts the risk of cardiovascular disease (CVD) based on patient health metrics. Built with Streamlit, Plotly, and Scikit-learn.

Features

1.  Live Risk Prediction

Interactive Form: Input patient details like age, weight, blood pressure, and cholesterol levels.

Real-time Machine Learning: Uses a pre-trained pipeline (cvd_best_pipeline.joblib) to calculate risk probability instantly.

Risk Classification: Classifies patients as Low Risk or High Risk with color-coded feedback.

Data Logging: Automatically logs every prediction to prediction_log.csv for future monitoring.

2.  Interactive Health Dashboard

Population Overview: Gauge charts and bar graphs showing CVD prevalence in the dataset.

Feature Analysis: Correlation heatmaps and feature importance charts to understand what drives the model.

Deep Dives:

Blood Pressure Categories: Visualizes risk across Normal, Elevated, and Hypertension stages.

BMI Analysis: Breaks down risk by Underweight, Normal, Overweight, and Obese categories.

Age vs. Risk: Trends showing how disease prevalence increases with age.

Tech Stack

Frontend: Streamlit

Visualization: Plotly Express & Graph Objects

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn, Joblib

Project Structure

```
├── artifacts/
│   ├── cvd_best_pipeline.joblib   # Trained ML Model Pipeline
│   └── scaler.joblib              # Scaler for normalizing input data
├── cleaned_data.csv               # Dataset used for dashboard visualizations
├── prediction_log.csv             # Auto-generated log of live predictions
├── requirements.txt               # List of dependencies
├── streamlit_app.py               # Main application file
└── README.md                      # Project documentation
```


Installation & Local Setup

Clone the repository:

```
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```


Create a virtual environment (Optional but Recommended):

```
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```


Install dependencies:

`pip install -r requirements.txt`


Run the application:

`python .\app.py`
`python .\api.py`


Access the App:
Open your browser and go to https://deploy-app-app-2ijtnwglh2wl2xv7yxtude.streamlit.app/
