import streamlit as st
import mlflow
import mlflow.sklearn
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset (to get the original label encoders)
ds_salaries = pd.read_csv('/Users/shiva/Desktop/IITJ/MLOps/DS Salaries/ds_salaries.csv')

# Encoding categorical features
columns_to_encode = [
    'experience_level', 'employment_type', 'job_title', 'employee_residence',
    'remote_ratio', 'company_location', 'company_size', 'salary_currency'
]

label_encoders = {}
encoded_values = {}

for col in columns_to_encode:
    le = LabelEncoder()
    ds_salaries[col + '_Encoded'] = le.fit_transform(ds_salaries[col])
    label_encoders[col] = le
    encoded_values[col] = list(le.classes_)

mapping_list = {}

for col in columns_to_encode:
    mapping_list[f"{col}_list"] = encoded_values[col]

def list_usage(col):
    if f"{col}_list" in mapping_list:
        return encoded_values[col]
    else:
        return []  # Return an empty list or handle the case where the list doesn't exist

# Load the best model's run_id
with open("best_model_run_id.json", "r") as f:
    best_model_info = json.load(f)

best_model_name = best_model_info["best_model_name"]
run_id = best_model_info["run_id"]

# Load the best model from MLflow
model_uri = f"runs:/{run_id}/{best_model_name.lower().replace(' ', '_')}_model"
best_model = mlflow.sklearn.load_model(model_uri)

# Streamlit app for salary prediction
st.title("Data Scientist Salary Prediction")
st.write("Enter the features to get a salary prediction:")

work_year = st.number_input("Work Year", min_value=2020, max_value=2024, value=2023)
experience_level = st.selectbox("Experience Level", list_usage('experience_level'))
employment_type = st.selectbox("Employment Type",  list_usage('employment_type'))
job_title = st.selectbox("Job Title", list_usage('job_title'))
salary_currency = st.selectbox("Salary Currency", list_usage('salary_currency'))
employee_residence = st.selectbox("Employee Residence", list_usage('employee_residence'))
remote_ratio = st.slider("Remote Ratio", 0, 100, 100)
company_location = st.selectbox("Company Location", list_usage('company_location'))
company_size = st.selectbox("Company Size", ['S', 'M', 'L'])

# Encoding user input using the same label encoders
def encode_input(value, encoder):
    return encoder.transform([value])[0]

encoded_input = {
    'work_year': work_year,
    'salary_currency_Encoded': encode_input(salary_currency, label_encoders['salary_currency']),
    'experience_level_Encoded': encode_input(experience_level, label_encoders['experience_level']),
    'employment_type_Encoded': encode_input(employment_type, label_encoders['employment_type']),
    'job_title_Encoded': encode_input(job_title, label_encoders['job_title']),
    'employee_residence_Encoded': encode_input(employee_residence, label_encoders['employee_residence']),
    'remote_ratio_Encoded': remote_ratio,
    'company_location_Encoded': encode_input(company_location, label_encoders['company_location']),
    'company_size_Encoded': encode_input(company_size, label_encoders['company_size']),
}

# Convert the input into a DataFrame
input_df = pd.DataFrame([encoded_input])

# Predict the salary
if st.button(f"Salary Prediction for {job_title} in {company_location} (Click Here)"):
    predicted_salary = best_model.predict(input_df)[0]
    st.write(f"Predicted Salary in USD : ${predicted_salary:.2f}")

# Display the performance metrics and model info
st.write(f"Model: {best_model_name}")
st.write(f"Run ID: {run_id}")

def display_performance_plots():
    image_path = f"optimized_{best_model_name.lower().replace(' ', '_')}_plot.png"
    st.image(image_path, caption=f"Actual vs Predicted for Optimized {best_model_name}", use_column_width=True)

display_performance_plots()