## Importing Libraries
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

## Data Loading
# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model_year', 'origin', 'car_name']
df = pd.read_csv(url, delim_whitespace=True, names=columns)

## Data Exploration
df.head()
# Step 2: Inspect the dataset
df.info()


df.describe()
df.isnull().sum()
df.horsepower.unique()
# Handle missing values in 'horsepower'
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')  # Convert '?' to NaN
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)
# Step 3: EDA
sns.pairplot(df, x_vars=['cylinders', 'displacement', 'horsepower', 'weight'], y_vars='mpg', height=4)
plt.title("Pairplot of Features with MPG")
plt.show()
df_numeric = df.select_dtypes(include=['number'])
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# **Preprocessing**
# Step 4: Preprocessing
# Drop 'car_name'
df.drop(columns=['car_name'], inplace=True)

# Define target and features
X = df.drop(columns=['mpg'])
y = df['mpg']

# Column transformer for scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['displacement', 'horsepower', 'weight', 'acceleration']),
        ('cat', OneHotEncoder(), ['origin'])
    ])

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# **Model Training**
# Step 1: Create a DataFrame to Store Results
results = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'RMSE': [],
    'MAE': []
}

# Collect RMSE and MAE from the trained models
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results['RMSE'].append(rmse)
    results['MAE'].append(mae)

# Add the tuned Random Forest results
results['Model'].append('Tuned Random Forest')
results['RMSE'].append(rmse)  # From grid search
results['MAE'].append(mae)  # From grid search

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
results_df

# Bar Plot for RMSE and MAE
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.barplot(data=results_df, x='Model', y='RMSE', palette='viridis')
plt.title('RMSE Comparison')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(data=results_df, x='Model', y='MAE', palette='viridis')
plt.title('MAE Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# Scatter Plot: Actual vs Predicted for the Best Model (Tuned Random Forest)
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.8)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs Predicted MPG')
plt.legend()
plt.show()

# Residual Plot for the Best Model
residuals = y_test - y_pred_best

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred_best, y=residuals, alpha=0.8)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Residual Plot for Tuned Random Forest')
plt.show()
# **Hyperparameter Tuning for Random Forest**
# Step 7: Hyperparameter Tuning for Random Forest
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(
    Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
    param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

# Best parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate the tuned model
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"Tuned Random Forest:\n\tRMSE: {rmse:.2f}\n\tMAE: {mae:.2f}")
# **Feature Importance**
# Step 8: Feature Importance
final_rf = grid_search.best_estimator_.named_steps['model']
feature_importance = final_rf.feature_importances_
feature_names = preprocessor.transformers_[0][2] + list(
    grid_search.best_estimator_.named_steps['preprocessor']
    .transformers_[1][1]
    .get_feature_names_out()
)
sns.barplot(x=feature_importance, y=feature_names)
plt.title("Feature Importance")
plt.show()
