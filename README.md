*This project was developed with the purpose of applying the most current MLOPS (Multi-Layer Operating Procedures) locally, for learning purposes.*

# Customer Churn Prediction MLOps

## Business Problem

Customer churn directly impacts company revenue and growth. This project aims to predict the probability of customer churn using machine learning and demonstrates the implementation of a production-ready MLOps workflow.

## Solution Overview

Raw Data
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
FastAPI Inference
↓
Monitoring

## Architecture

data/
 ├─ bronze
 ├─ silver
 └─ gold

src/
 ├─ ingestion
 ├─ validation
 ├─ features
 ├─ training
 ├─ inference
 └─ utils

## Project Structure

customer-churn-mlops/
├── data/
├── src/
├── tests/
├── models/
├── configs/
└── ...

## Tech Stack

| Tool         | Purpose          |
| ------------ | ---------------- |
| Python       | Development      |
| Pandas       | Data processing  |
| Scikit-Learn | Machine learning |
| Ruff         | Linting          |
| Black        | Formatting       |
| Pytest       | Testing          |


## Setup

git clone <repo>

cd customer-churn-mlops

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

## Running the Project

Project setup completed.

Data pipeline implementation is currently under development.

## MLOps Roadmap

- [x] Project structure
- [x] Environment setup
- [x] Data ingestion
- [x] Data validation
- [x] Feature engineering
- [x] Training pipeline
- [x] MLflow
- [x] FastAPI
- [ ] CI/CD
- [ ] Monitoring
- [ ] Deployment
- [ ] Docker

## Future Improvements

- Hyperparameter optimization
- Automated retraining
- Feature store integration
- Cloud deployment
- Drift monitoring
```
churn-mlops
├─ .pytest_cache
│  ├─ CACHEDIR.TAG
│  ├─ README.md
│  └─ v
│     └─ cache
│        └─ nodeids
├─ .ruff_cache
│  ├─ 0.15.15
│  │  ├─ 15156875728144846687
│  │  ├─ 1623204139137528218
│  │  ├─ 2520914726518562419
│  │  ├─ 7473881215035151206
│  │  └─ 7805723845045947902
│  └─ CACHEDIR.TAG
├─ configs
│  └─ config.yaml
├─ data
├─ models
├─ notebooks
│  └─ exploratory_data_analysis
│     └─ eda.ipynb
├─ pyproject.toml
├─ README.md
├─ requirements.txt
├─ src
│  ├─ features
│  │  ├─ build_features.py
│  │  └─ __init__.py
│  ├─ inference
│  │  ├─ api.py
│  │  ├─ predictor.py
│  │  └─ __init__.py
│  ├─ ingestion
│  │  ├─ ingest.py
│  │  └─ __init__.py
│  ├─ training
│  │  ├─ evalueate.py
│  │  ├─ preprocessing.py
│  │  ├─ train.py
│  │  └─ __init__.py
│  ├─ utils
│  │  ├─ config.py
│  │  └─ __init__.py
│  ├─ validation
│  │  ├─ validate.py
│  │  └─ __init__.py
│  └─ __init__.py
└─ tests
   └─ test_config.py

```