
#  Human Activity Recognition (HAR) App

A real-time Android application that detects human activities like walking, running, sitting, standing, and laying using smartphone sensors and a machine learning model.

---

##  Features

-  Real-time activity detection
- Uses accelerometer + gyroscope sensors
- Machine Learning model (Random Forest)
-  Backend API using FastAPI
- Live data streaming from mobile to server

---

## How It Works

1. The Android app collects sensor data (accelerometer + gyroscope)
2. Data is sent to a backend server via HTTP
3. Backend extracts features and runs ML model
4. Predicted activity is sent back to the app
5. App displays activity with icon

---

##  Tech Stack

**Frontend (Android)**
- Kotlin
- XML UI

**Backend**
- FastAPI (Python)
- NumPy, Joblib

**Machine Learning**
- Scikit-learn (Random Forest)

---

 API Endpoint
