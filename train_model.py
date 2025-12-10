import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load dataset
df = pd.read_csv("breast_cancer_data.csv")

# Clean unnecessary columns
df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

# Target
y = df["diagnosis"].map({"M": 1, "B": 0})

# Features
X = df.drop(columns=["id", "diagnosis"], errors="ignore")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train AdaBoost
model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save MODEL
os.makedirs("Weight files", exist_ok=True)
with open("Weight files/adaboost_model_with_smote_on_original_data.pkl", "wb") as f:
    pickle.dump(model, f)

# Save SCALER
with open("Weight files/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✔ Model and scaler saved successfully!")
