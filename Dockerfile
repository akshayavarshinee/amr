FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY params.yaml run_api.py run_phase1.py ./

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "run_api.py"]
