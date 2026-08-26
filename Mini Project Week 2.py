# Import libraries for Mini Project 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
import zipfile
import pandas as pd

# The exact path to your zip file on the D drive
zip_path = r"D:\ML_Week2_proj\archive.zip"

# Open the zip file and read Housing.csv directly from inside it
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open('Housing.csv') as f:
        df = pd.read_csv(f)

print("Dataset loaded successfully!")
print(df.head())

# Select features and target for the Mini Project
# Checking available columns and filtering numerical features for the model
X = df.select_dtypes(include=[np.number]).drop(columns=['price'], errors='ignore')
y = df['price']

# Handle missing data if present
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=['price'])
y = data['price']

# Train Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 1: Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Task 2: Predict house prices
y_pred = lr_model.predict(X_test)

# Task 3: Evaluate R^2 score (and additional metrics for completeness)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== Mini Project 2 Results ===")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Task 4: Plot predicted vs actual values
plt.figure(figsize=(9, 6))
plt.scatter(y_test, y_pred, color='teal', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='orange', linestyle='--', lw=2, label='Ideal Fit')
plt.xlabel("Actual House Prices", fontsize=12)
plt.ylabel("Predicted House Prices", fontsize=12)
plt.title("Mini Project 2: Actual vs Predicted House Prices", fontsize=14)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
