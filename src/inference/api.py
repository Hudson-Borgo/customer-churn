from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predictor import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    version="0.1.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/predict")
def predict(customer: CustomerData):
    result = predict_churn(customer.model_dump())
    return result
