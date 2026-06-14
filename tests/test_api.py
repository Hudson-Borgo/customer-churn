from unittest.mock import patch

from fastapi.testclient import TestClient

from src.inference.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 99.5,
        "TotalCharges": 199.0,
    }

    with patch(
        "src.inference.api.predict_churn",
        return_value={
            "prediction": 1,
            "churn_probability": 0.8,
        },
    ):
        response = client.post("/predict", json=payload)

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["prediction"] == 1
    assert response_data["churn_probability"] == 0.8
