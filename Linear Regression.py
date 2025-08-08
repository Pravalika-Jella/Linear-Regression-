# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 2. Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset shape:", df.shape)
print(df.head())

# --- SIMPLE LINEAR REGRESSION (One feature: MedInc) ---
X = df[['MedInc']]   # Median income
y = df['MedHouseVal']  # Median house value

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
lr_simple = LinearRegression()
lr_simple.fit(X_train, y_train)

# Predict
y_pred_simple = lr_simple.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred_simple)
mse = mean_squared_error(y_test, y_pred_simple)
r2 = r2_score(y_test, y_pred_simple)

print("\n--- Simple Linear Regression ---")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)
print("Coefficient (slope):", lr_simple.coef_[0])
print("Intercept:", lr_simple.intercept_)

# Plot
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_simple, color='red', linewidth=2, label='Predicted')
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# --- MULTIPLE LINEAR REGRESSION (All features) ---
X_multi = df.drop('MedHouseVal', axis=1)
y_multi = df['MedHouseVal']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

lr_multi = LinearRegression()
lr_multi.fit(X_train_m, y_train_m)

y_pred_multi = lr_multi.predict(X_test_m)

# Metrics
mae_m = mean_absolute_error(y_test_m, y_pred_multi)
mse_m = mean_squared_error(y_test_m, y_pred_multi)
r2_m = r2_score(y_test_m, y_pred_multi)

print("\n--- Multiple Linear Regression ---")
print("MAE:", mae_m)
print("MSE:", mse_m)
print("R² Score:", r2_m)
print("Coefficients:", lr_multi.coef_)
print("Intercept:", lr_multi.intercept_)

# Scatter actual vs predicted
plt.scatter(y_test_m, y_pred_multi, color='purple')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.show()

# --- VIF Calculation for Multicollinearity ---
vif_data = pd.DataFrame()
vif_data["Feature"] = X_multi.columns
vif_data["VIF"] = [variance_inflation_factor(X_multi.values, i) for i in range(X_multi.shape[1])]

print("\n--- Variance Inflation Factor (VIF) ---")
print(vif_data)