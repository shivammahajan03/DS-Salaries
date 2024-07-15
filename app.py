import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

# Load the dataset
ds_salaries = pd.read_csv('/Users/shiva/Desktop/IITJ/MLOps/DS Salaries/ds_salaries.csv')

# Encode categorical features
columns_to_encode = [
    'experience_level', 'employment_type', 'job_title', 'employee_residence',
    'remote_ratio', 'company_location', 'company_size', 'salary_currency'
]

label_encoders = {}
for col in columns_to_encode:
    le = LabelEncoder()
    ds_salaries[col + '_Encoded'] = le.fit_transform(ds_salaries[col])
    label_encoders[col] = le

# Columns to keep in the new DataFrame
columns_to_keep = [
    'work_year', 'salary', 'salary_currency_Encoded', 'salary_in_usd',
    'experience_level_Encoded', 'employment_type_Encoded', 'job_title_Encoded',
    'employee_residence_Encoded', 'remote_ratio_Encoded', 'company_location_Encoded',
    'company_size_Encoded'
]

new_df = ds_salaries[columns_to_keep]

# Features and target
X = new_df.drop('salary_in_usd', axis=1)
y = new_df['salary_in_usd']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the best model
model_uri = "runs:/59e294a8864042adae195b038733eed9/" 
best_model = mlflow.sklearn.load_model(model_uri)

# Streamlit App
st.title('Data Scientist Salary Prediction')

# Sidebar for user input
st.sidebar.header('User Input Features')

def user_input_features():
    work_year = st.sidebar.number_input("Work Year", min_value=2020, max_value=2024, step=1)
    salary_currency = st.sidebar.selectbox("Salary Currency", ds_salaries['salary_currency'].unique())
    experience_level = st.sidebar.selectbox("Experience Level", ds_salaries['experience_level'].unique())
    employment_type = st.sidebar.selectbox("Employment Type", ds_salaries['employment_type'].unique())
    job_title = st.sidebar.selectbox("Job Title", ds_salaries['job_title'].unique())
    employee_residence = st.sidebar.selectbox("Employee Residence", ds_salaries['employee_residence'].unique())
    remote_ratio = st.sidebar.selectbox("Remote Ratio", ds_salaries['remote_ratio'].unique())
    company_location = st.sidebar.selectbox("Company Location", ds_salaries['company_location'].unique())
    company_size = st.sidebar.selectbox("Company Size", ds_salaries['company_size'].unique())

    data = {
        'work_year': work_year,
        'salary_currency': label_encoders['salary_currency'].transform([salary_currency])[0],
        'experience_level': label_encoders['experience_level'].transform([experience_level])[0],
        'employment_type': label_encoders['employment_type'].transform([employment_type])[0],
        'job_title': label_encoders['job_title'].transform([job_title])[0],
        'employee_residence': label_encoders['employee_residence'].transform([employee_residence])[0],
        'remote_ratio': label_encoders['remote_ratio'].transform([remote_ratio])[0],
        'company_location': label_encoders['company_location'].transform([company_location])[0],
        'company_size': label_encoders['company_size'].transform([company_size])[0]
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combine user input features with the entire dataset
# This will be useful for encoding the user input features
ds_salaries = ds_salaries.drop(columns=['salary_in_usd'])
df = pd.concat([input_df, ds_salaries], axis=0)

# Encoding categorical features
for col in columns_to_encode:
    df[col + '_Encoded'] = label_encoders[col].transform(df[col])

# Selects only the first row (the user input data)
df = df[:1]

# Display user input
st.subheader('User Input features')
st.write(df)

# Make prediction
prediction = best_model.predict(df)

st.subheader('Prediction')
st.write(f"Predicted Salary in USD: ${prediction[0]:.2f}")

# Display metrics
st.subheader('Model Performance Metrics')
rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
mae = mean_absolute_error(y_test, best_model.predict(X_test))
r2 = r2_score(y_test, best_model.predict(X_test))
st.write(f"RMSE: {rmse:.4f}")
st.write(f"MAE: {mae:.4f}")
st.write(f"R²: {r2:.4f}")

# Visualize model performance
st.subheader('Model Performance Visualization')

# Plotting actual vs predicted values for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_model.predict(X_test), alpha=0.5)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Optimized Best Model")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
st.pyplot(plt)
