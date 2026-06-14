# Customer Churn MLOps

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![CI](https://github.com/Hudson-Borgo/customer-churn/actions/workflows/ci.yml/badge.svg)

End-to-end MLOps project for customer churn prediction using Python, Scikit-Learn, MLflow, FastAPI, Docker, CI/CD, and monitoring concepts.

This project was built for learning and practicing modern MLOps and ML Engineering concepts using a production-inspired architecture.

---

## Learning Purpose

This is a personal project focused on learning how real-world machine learning systems are structured, deployed, monitored, and maintained.

The main goal is not achieving the best churn prediction model, but rather implementing a complete and reproducible ML workflow following engineering and MLOps best practices.

The project focuses on:

- reproducible pipelines
- modular project architecture
- experiment tracking
- inference APIs
- Docker containerization
- CI/CD workflows
- monitoring foundations
- automated quality checks

---

# Business Problem

Customer churn directly impacts revenue, retention, and company growth.

This project predicts the probability of customer churn using machine learning while simulating how a production-oriented ML system would be implemented.

---

# Solution Architecture

```text
Raw Data
    ↓
Data Ingestion
    ↓
Data Validation
    ↓
Feature Engineering
    ↓
Model Training
    ↓
MLflow Tracking
    ↓
Model Registry
    ↓
FastAPI Inference API
    ↓
Prediction Logging
    ↓
Drift Monitoring
    ↓
Docker Deployment
```

---

# Project Structure

```text
customer-churn/
│
├── artifacts/
│   └── model.pkl
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── logs/
│
├── notebooks/
│   └── exploratory_data_analysis/
│
├── src/
│   ├── features/
│   ├── inference/
│   ├── ingestion/
│   ├── monitoring/
│   ├── training/
│   ├── utils/
│   └── validation/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| Scikit-Learn | Machine learning |
| MLflow | Experiment tracking |
| FastAPI | Inference API |
| Docker | Containerization |
| Docker Compose | Service orchestration |
| Pytest | Automated testing |
| Ruff | Linting |
| Black | Code formatting |
| GitHub Actions | CI/CD |

---

# MLOps Concepts Implemented

## Data Engineering

- Bronze / Silver / Gold architecture
- Reproducible pipelines
- Data validation

## Machine Learning

- Feature engineering
- Train / validation / test split
- Logistic Regression baseline model
- Model artifact generation

## Experiment Tracking

- MLflow experiments
- Parameter logging
- Metric logging
- Model logging

## Inference

- FastAPI prediction endpoint
- Health check endpoint
- JSON request validation

## Monitoring

- Prediction logging
- Drift detection foundations
- Inference observability

## Software Engineering

- Modular architecture
- Automated tests
- Linting and formatting
- Makefile automation

## DevOps / MLOps

- Docker containerization
- Docker Compose
- CI/CD with GitHub Actions

---

# Local Setup

## Clone repository

```bash
git clone https://github.com/Hudson-Borgo/customer-churn.git

cd customer-churn
```

---

## Create virtual environment

### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

## Run full pipeline

```bash
make pipeline
```

---

## Individual stages

```bash
make ingest
make validate
make features
make train
```

---

# MLflow

## Start MLflow UI

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

# Running the API

## Local execution

```bash
make api
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# Docker

## Build container

```bash
make docker-build
```

---

## Run with Docker Compose

```bash
make docker-up
```

Open:

```text
http://127.0.0.1:8000/docs
```

Stop containers:

```bash
make docker-down
```

---

# API Example

## POST `/predict`

Request:

```json
{
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
  "TotalCharges": 199.0
}
```

Response:

```json
{
  "prediction": 1,
  "churn_probability": 0.81
}
```

---

# Automated Tests

Run tests:

```bash
make test
```

Run quality checks:

```bash
make quality
```

---

# CI/CD

GitHub Actions automatically runs:

- Ruff
- Black
- Pytest

on every push and pull request.

---

# Monitoring

Current monitoring features:

- prediction logging
- inference tracking
- drift detection foundations

Future improvements:

- PSI drift metrics
- model performance monitoring
- Prometheus/Grafana integration
- Evidently AI integration

---

# Future Improvements

- Hyperparameter tuning
- Feature Store integration
- Cloud deployment
- Model Registry improvements
- Advanced monitoring
- Kubernetes deployment
- Canary deployments
- Automated retraining

---

# Key Engineering Concepts Demonstrated

This project demonstrates concepts commonly used in ML Engineering and MLOps environments:

- modular ML systems
- reproducible pipelines
- containerized inference
- CI/CD workflows
- experiment tracking
- API serving
- monitoring foundations
- software engineering practices for ML

---


## MLflow

![MLflow](assets/mlflow.png)

## FastAPI Swagger

![Swagger](assets/swagger.png)

## Docker Container

![Docker](assets/docker.png) 

# Author

Hudson Borgo

Senior Data Scientist focused on Machine Learning Engineering, MLOps, and production-oriented AI systems.