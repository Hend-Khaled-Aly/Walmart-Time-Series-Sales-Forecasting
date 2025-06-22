from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from model import load_model
from predict import make_prediction

app = FastAPI()
model = load_model()

class PredictRequest(BaseModel):
    sequence: List[List[float]]  # shape: [seq_len, num_features]

@app.post("/predict")
def predict(req: PredictRequest):
    prediction = make_prediction(model, req.sequence)
    return {"prediction": prediction.tolist()}
