.PHONY: lint test run docker-build

lint:
    flake8 src/ --max-line-length=120

test:
    pytest tests/

run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

docker-build:
    docker build -t creditrisk-api .

docker-run:
    docker run -p 8000:8000 creditrisk-api