import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
DATA_PATH = "lung cancer data.csv"

if not os.path.exists(DATA_PATH):
    print("⚠️ Dataset not found. Please download from Kaggle and place it in the project folder.")
    exit()

df = pd.read_csv(DATA_PATH)

# Encode categorical variables
label_encoders = {}
categorical_columns = ["GENDER", "SMOKING", "COUGHING", "FATIGUE", "SHORTNESS_BREATH", "LUNG_CANCER"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split dataset
X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "lung_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully!")

