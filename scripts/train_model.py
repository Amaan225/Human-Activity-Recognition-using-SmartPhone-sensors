import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X = pd.read_csv("data/UCI HAR Dataset/train/X_train.txt", sep=r"\s+", header=None)
y = pd.read_csv("data/UCI HAR Dataset/train/y_train.txt", header=None)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Save model
joblib.dump(model, "backend/har_model.pkl")
print("Model saved to backend/har_model.pkl")

