.PHONY: install test train serve docker-build

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

train:
	python -m fraudguard.train --output artifacts/fraudguard.joblib

serve:
	python -m uvicorn fraudguard.api:app --reload

docker-build:
	docker build -t fraudguard:local .
