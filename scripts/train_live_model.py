import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# Paths
X_PATH = "processed_data/X.npy"
y_PATH = "processed_data/y.npy"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X = np.load(X_PATH)
y = np.load(y_PATH)

print("Loaded:", X.shape, y.shape)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline (VERY IMPORTANT)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    ))
])

# Train
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# Save model
MODEL_PATH = os.path.join(MODEL_DIR, "har_live_model.pkl")
joblib.dump(pipe, MODEL_PATH)

print("Model saved to:", MODEL_PATH)
