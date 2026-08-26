# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the Titanic dataset safely using the exact path
import pandas as pd

# Just copy-paste this full path with the 'r' in front of it
df = pd.read_csv(r"D:\ML_Week2_proj\Titanic-Dataset.csv.csv")

print("Dataset loaded successfully!")
print(df.head())

print("Dataset Head:")
print(df.head())

# 2. Data Preprocessing & Feature Selection
# Select important columns for survival prediction
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
df_subset = df[features].copy()

# Handle missing values (fill missing Age with the median)
df_subset['Age'] = df_subset['Age'].fillna(df_subset['Age'].median())
df_subset.dropna(inplace=True)

# Convert categorical 'Sex' column into numeric values (male: 0, female: 1)
df_subset['Sex'] = df_subset['Sex'].map({'male': 0, 'female': 1})

# Separate features (X) and target variable (y)
X = df_subset.drop('Survived', axis=1)
y = df_subset['Survived']

# 3. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training (Logistic Regression)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 5. Prediction and Evaluation
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Confusion Matrix Plot
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Titanic Survival - Confusion Matrix")
plt.show()
