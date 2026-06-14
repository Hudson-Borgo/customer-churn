from unittest.mock import Mock, patch

from src.inference.predictor import predict_churn


def test_predict_churn_returns_expected_keys():
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

    mock_model = Mock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]

    with (
        patch("src.inference.predictor.load_model", return_value=mock_model),
        patch("src.inference.predictor.log_prediction"),
    ):
        result = predict_churn(payload)

    assert "prediction" in result
    assert "churn_probability" in result


def test_predict_churn_returns_valid_types():
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

    mock_model = Mock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]

    with (
        patch("src.inference.predictor.load_model", return_value=mock_model),
        patch("src.inference.predictor.log_prediction"),
    ):
        result = predict_churn(payload)

    assert isinstance(result["prediction"], int)
    assert isinstance(result["churn_probability"], float)
