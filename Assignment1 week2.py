# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
import pandas as pd
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

print("Dataset Head:")
print(df.head())

# Display basic information and first few rows
print("Dataset Head:")
print(df.head())

# 2. Data Preprocessing (Handling numerical/categorical conversion if necessary)
# Assuming standard columns like 'price', 'area', 'bedrooms', 'bathrooms' exist
# Let's select a few key features for prediction, or handle mapping if categorical columns are present.
# For a standard generic approach, let's look at numerical columns:
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical Columns:", numerical_cols)

# Define Features (X) and Target (y) - assuming 'price' is the target column
target = 'price'
features = [col for col in numerical_cols if col != target]

X = df[features]
y = df[target]

# Drop missing values if any
data = pd.concat([X, y], axis=1).dropna()
X = data[features]
y = data[target]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction and Evaluation
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 6. Plotting Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Assignment 1)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.show()
