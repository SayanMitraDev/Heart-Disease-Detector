# Heart Disease Detection using Machine Learning

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (replace with your file path)
# Dataset should have a 'target' column (0 = no disease, 1 = disease)
data = pd.read_csv("heart.csv")

# Show first rows
print(data.head())

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---- Prediction on new data ----
# Example input (must match dataset feature order)
sample = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("\n⚠️ Heart Disease Detected")
else:
    print("\n✅ No Heart Disease")