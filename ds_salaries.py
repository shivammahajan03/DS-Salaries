# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Load the dataset
ds_salaries = pd.read_csv('/Users/shiva/Desktop/IITJ/MLOps/DS Salaries/ds_salaries.csv')

# Display the first few rows of the DataFrame
print("First few rows of the dataset:")
print(ds_salaries.head())

# Display summary statistics
print("\nSummary statistics:")
print(ds_salaries.describe())

# Check for missing values
print("\nMissing values in each column:")
print(ds_salaries.isnull().sum())

# Encoding categorical features
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

# Deciding the features and target
X = new_df.drop('salary_in_usd', axis=1)
y = new_df['salary_in_usd']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLFlow Logging: Start experiment
mlflow.set_experiment("DS Salaries")

with mlflow.start_run():
    # Train the Ridge Model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)

    # Log parameters and metrics for Ridge Regression
    mlflow.log_param("ridge_alpha", 1.0)
    mlflow.log_metric("ridge_rmse", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
    mlflow.log_metric("ridge_r2", r2_score(y_test, y_pred_ridge))

    # Log the Ridge model
    mlflow.sklearn.log_model(ridge_model, "ridge_model")

    # Train the Lasso Model
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)

    # Log parameters and metrics for Lasso Regression
    mlflow.log_param("lasso_alpha", 0.1)
    mlflow.log_metric("lasso_rmse", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
    mlflow.log_metric("lasso_r2", r2_score(y_test, y_pred_lasso))

    # Log the Lasso model
    mlflow.sklearn.log_model(lasso_model, "lasso_model")

    # Function for calculating the metrics of models
    def print_metrics(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"{model_name} - RMSE: {rmse:.4f}, R_squared: {r2:.4f}")

    # Print the performance metrics for both models
    print_metrics(y_test, y_pred_ridge, "Ridge Regression Model")
    print_metrics(y_test, y_pred_lasso, "Lasso Regression Model")

    # Plotting actual vs predicted values for both models
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_ridge, alpha=0.5)
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Ridge Regression")
    plt.savefig("ridge_regression_plot.png")

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_lasso, alpha=0.5)
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Lasso Regression")
    plt.savefig("lasso_regression_plot.png")

    mlflow.log_artifact("ridge_regression_plot.png")
    mlflow.log_artifact("lasso_regression_plot.png")

    plt.tight_layout()
    plt.show()
