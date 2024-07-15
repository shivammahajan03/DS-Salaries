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
import json

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
encoded_values = {}

for col in columns_to_encode:
    le = LabelEncoder()
    ds_salaries[col + '_Encoded'] = le.fit_transform(ds_salaries[col])
    label_encoders[col] = le
    encoded_values[col] = list(le.classes_)

mapping_list = {}

for col in columns_to_encode:
    mapping_list[f"{col}_list"] = encoded_values[col]

# Columns to keep in the new DataFrame
columns_to_keep = [
    'work_year', 'salary_currency_Encoded', 'salary_in_usd',
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

# Function to train and evaluate a model
def train_evaluate_model(model, model_name):
    with mlflow.start_run(nested=True, run_name=model_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        mlflow.log_param(f"{model_name}_params", model.get_params())
        mlflow.log_metric(f"{model_name}_rmse", rmse)
        mlflow.log_metric(f"{model_name}_mae", mae)
        mlflow.log_metric(f"{model_name}_r2", r2)
        
        mlflow.sklearn.log_model(model, f"{model_name.lower().replace(' ', '_')}_model")
        
        return y_pred, rmse, run.info.run_id

# MLFlow Logging: Start experiment
mlflow.set_experiment("DS Salaries Log")

best_run_id = None

with mlflow.start_run(run_name="Overall Experiment") as parent_run:
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100)
    }
    
    predictions = {}
    rmses = {}
    run_ids = {}
    
    for model_name, model in models.items():
        print(f"\nTraining and evaluating {model_name}...")
        predictions[model_name], rmse, run_id = train_evaluate_model(model, model_name)
        rmses[model_name] = rmse
        run_ids[model_name] = run_id

    # Plotting actual vs predicted values for all models
    plt.figure(figsize=(18, 10))
    
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        plt.subplot(2, 3, i + 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Salary")
        plt.ylabel("Predicted Salary")
        plt.title(f"{model_name}")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    
    plt.tight_layout()
    plt.savefig("models_comparison_plot.png")
    mlflow.log_artifact("models_comparison_plot.png")
    plt.show()

# Find the best performing model
best_model_name = min(rmses, key=rmses.get)
best_run_id = run_ids[best_model_name]
print(f"\nBest performing model: {best_model_name} with RMSE: {rmses[best_model_name]:.4f}")

# Save the best model's run_id to a file
with open("best_model_run_id.json", "w") as f:
    json.dump({"best_model_name": best_model_name, "run_id": best_run_id}, f)

# Hyperparameter tuning for the best model
param_grids = {
    "Ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "Decision Tree": {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 10, 20]},
    "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
    "Gradient Boosting": {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.1, 0.2]}
}

best_model = models[best_model_name]
param_grid = param_grids[best_model_name]

grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Log the best parameters and the best score
with mlflow.start_run(run_id=best_run_id, nested=True):
    mlflow.log_param(f"{best_model_name}_best_params", grid_search.best_params_)
    mlflow.log_metric(f"{best_model_name}_best_score", -grid_search.best_score_)

print(f"\nBest parameters for {best_model_name}: {grid_search.best_params_}")
print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")

# Evaluate the best model with the best parameters on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nOptimized {best_model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

with mlflow.start_run(run_id=best_run_id, nested=True):
    mlflow.log_metric(f"{best_model_name}_optimized_rmse", rmse)
    mlflow.log_metric(f"{best_model_name}_optimized_mae", mae)
    mlflow.log_metric(f"{best_model_name}_optimized_r2", r2)

# Plotting actual vs predicted values for the optimized best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title(f"Optimized {best_model_name}")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.savefig(f"optimized_{best_model_name.lower().replace(' ', '_')}_plot.png")
with mlflow.start_run(run_id=best_run_id, nested=True):
    mlflow.log_artifact(f"optimized_{best_model_name.lower().replace(' ', '_')}_plot.png")
plt.show()

# Register the best model in the MLflow Model Registry
model_uri = f"runs:/{best_run_id}/{best_model_name.lower().replace(' ', '_')}_model"
model_details = mlflow.register_model(model_uri, best_model_name)

# Transition the model to the "Production" stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

# Save the registered model details to a JSON file
with open("registered_model_details.json", "w") as f:
    json.dump({
        "model_name": model_details.name,
        "model_version": model_details.version
    }, f)
