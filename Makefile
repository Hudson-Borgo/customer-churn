install:
	pip install -r requirements.txt

lint:
	ruff check .

format:
	black .

format-check:
	black --check .

test:
	pytest

quality:
	ruff check .
	black --check .
	pytest

ingest:
	python -m src.ingestion.ingest

validate:
	python -m src.validation.validate

features:
	python -m src.features.build_features

train:
	python -m src.training.train

api:
	uvicorn src.inference.api:app --reload

drift:
	python -m src.monitoring.drift

docker-build:
	docker build -t churn-api .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

pipeline:
	python -m src.ingestion.ingest
	python -m src.validation.validate
	python -m src.features.build_features
	python -m src.training.train
