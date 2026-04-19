from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import os

from features.fusion_features import extract_fusion_features

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "har_live_model.pkl")

model = joblib.load(MODEL_PATH)

class FusionInput(BaseModel):
    data: List[List[float]]  

LABEL_MAP = {
    0: "STANDING",
    1: "SITTING",
    2: "LAYING",
    3: "WALKING",
    4: "RUNNING"
}

@app.post("/predict_live")
def predict_live(data: FusionInput):
    X = extract_fusion_features(data.data)
    print("DEBUG X shape:", X.shape)  
    pred = int(model.predict(X)[0])
    return {"activity": LABEL_MAP[pred]}

