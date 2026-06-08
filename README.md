# Customer Churn Prediction — End-to-End MLOps Project

## Overview

This is a personal project created for learning and practicing modern MLOps concepts and machine learning engineering workflows.

The main objective is to simulate how a production-oriented ML system is structured, developed, validated, and maintained using industry-inspired practices.

The project covers:

* Data ingestion
* Data validation
* Feature engineering
* Machine learning training pipelines
* Experiment tracking with MLflow
* Model artifact generation
* FastAPI inference service
* CI pipeline with GitHub Actions

---

# Business Problem

Customer churn directly impacts company revenue and long-term growth.

Companies that can identify customers with a high probability of churn are able to:

* improve customer retention
* reduce revenue loss
* optimize retention campaigns
* allocate commercial efforts more efficiently

This project predicts customer churn probability using supervised machine learning techniques.

---

# Solution Architecture

```text
Raw CSV Dataset
        │
        ▼
Bronze Layer
(raw ingestion)
        │
        ▼
Silver Layer
(validation and data quality)
        │
        ▼
Gold Layer
(feature engineering)
        │
        ▼
Training Pipeline
(scikit-learn pipeline)
        │
        ▼
MLflow Tracking
(experiments and metrics)
        │
        ▼
Model Artifact
(model.pkl)
        │
        ▼
FastAPI Inference Service
        │
        ▼
Prediction Endpoint
```

---

# Data Architecture

The project follows the Bronze / Silver / Gold layered architecture 

## Bronze Layer

Raw immutable dataset.

Responsibilities:

* preserve source data


## Silver Layer

Validateddataset.

Responsibilities:

* schema validation
* missing value handling
* duplicate validation
* business rule enforcement

## Gold Layer

Machine-learning-ready dataset.

Responsibilities:

* feature preparation
* target creation
* final training schema

---

# Project Structure

```text
customer-churn/

├── configs/
│   └── config.yaml
│
├── notebooks/
│   └── exploratory_data_analysis/
│
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── features/
│   ├── training/
│   ├── inference/
│   └── utils/
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── Dockerfile
```

---

# Tech Stack

| Technology     | Purpose                   |
| -------------- | ------------------------- |
| Python         | Main programming language |
| Pandas         | Data manipulation         |
| Scikit-Learn   | Machine learning          |
| MLflow         | Experiment tracking       |
| FastAPI        | Inference API             |
| Ruff           | Linting                   |
| Black          | Code formatting           |
| Pytest         | Automated testing         |
| GitHub Actions | CI pipeline               |

---

# Machine Learning Pipeline

The training pipeline uses:

* `ColumnTransformer`
* `OneHotEncoder`
* `StandardScaler`
* `LogisticRegression`

The preprocessing and model are encapsulated into a single artifact, this guarantee consistency 

---

# MLflow Tracking

MLflow is used to track:

* model parameters
* metrics
* experiments
* trained model artifacts

Tracked examples:

* model type
* random state
* train/test split
* accuracy metrics

---

# FastAPI Inference Service

The project exposes a REST API for inference.

## Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Prediction Endpoint

```http
POST /predict
```

Example response:

```json
{
  "prediction": 1,
  "churn_probability": 0.6252
}
```

---

# Continuous Integration

GitHub Actions is configured to automatically run:

* Ruff
* Black
* Pytest

on every push and pull request to the `main` branch.

This ensures:

* code quality
* formatting consistency
* basic pipeline validation

---

# Setup

## Clone Repository

```bash
git clone https://github.com/Hudson-Borgo/customer-churn.git

cd customer-churn
```

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Data Ingestion

```bash
python -m src.ingestion.ingest
```

## Data Validation

```bash
python -m src.validation.validate
```

## Feature Engineering

```bash
python -m src.features.build_features
```

## Model Training

```bash
python -m src.training.train
```

## Start MLflow UI

```bash
mlflow ui
```

## Start FastAPI Service

```bash
uvicorn src.inference.api:app --reload
```

---

# Current Status

## Completed

* Project structure
* Data ingestion pipeline
* Data validation layer
* Feature engineering pipeline
* ML training pipeline
* MLflow integration
* FastAPI inference service
* CI pipeline with GitHub Actions

## In Progress

* Monitoring
* Drift detection
* Deployment strategy
* Docker support

---

# Future Improvements

* Hyperparameter optimization
* Model registry integration
* Data drift monitoring
* Model drift monitoring
* Automated retraining
* Cloud deployment
* Docker containerization
* Feature store integration

---

# Notes

Docker support was intentionally deferred because the current development environment does not provide administrator privileges.

The project architecture remains container-ready and deployment-oriented.