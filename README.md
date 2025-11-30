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

`├── artifacts/
│   ├── cvd_best_pipeline.joblib   # Trained ML Model Pipeline
│   └── scaler.joblib              # Scaler for normalizing input data
├── cleaned_data.csv               # Dataset used for dashboard visualizations
├── prediction_log.csv             # Auto-generated log of live predictions
├── requirements.txt               # List of dependencies
├── streamlit_app.py               # Main application file
└── README.md                      # Project documentation`


Installation & Local Setup

Clone the repository:

`git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name`


Create a virtual environment (Optional but Recommended):

`python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate`


Install dependencies:

`pip install -r requirements.txt`


Run the application:

`streamlit run streamlit_app.py`


Access the App:
Open your browser and go to http://localhost:8501

Deployment (Streamlit Cloud)

This app is optimized for Streamlit Cloud.

Push your code to GitHub.

Go to share.streamlit.io.

Click "New App".

Select your repository and branch.

Set Main file path to streamlit_app.py.

Click Deploy!

🧪 Model Details

The machine learning model was trained on a cardiovascular disease dataset. It uses a pipeline that includes:

Preprocessing: Scaling of continuous variables (Age, Height, Weight, Blood Pressure).

Feature Engineering: Calculation of BMI, Pulse Pressure, and Health Index.

Classification: A trained classifier (e.g., Random Forest/XGBoost) optimized for accuracy and recall.

📝 License

This project is open-source and available for educational and research purposes.