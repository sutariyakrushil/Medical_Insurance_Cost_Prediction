# Medical-Insurance-Cost-Prediction
Medical Insurance Cost Prediction
A simple Streamlit app that estimates annual medical insurance charges using a trained machine-learning model.
Features
- Enter age, sex, BMI, number of children, smoking status, and region.
- Predict estimated annual insurance charges.
- Compare the prediction with reference values in a bar chart.
Requirements
Python 3.9+ recommended.

Install the required packages:
pip install streamlit pandas numpy joblib scikit-learn
Run the App
Keep app.py, insurance_model.pkl, and model_features.pkl in the same folder, then run:
streamlit run app.py
Files
- app.py — Streamlit application.
- insurance_model.pkl — trained model.
- model_features.pkl — model input feature names.
- medical_insurance.csv — dataset.
- project.ipynb — notebook for project exploration/model development.
Note
Predictions are estimates for learning/demo purposes only, not professional insurance or medical advice.
