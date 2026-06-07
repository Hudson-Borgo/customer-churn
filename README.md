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
- [ ] Docker
- [ ] FastAPI
- [ ] CI/CD
- [ ] Monitoring
- [ ] Deployment

## Future Improvements

- Hyperparameter optimization
- Automated retraining
- Feature store integration
- Cloud deployment
- Drift monitoring