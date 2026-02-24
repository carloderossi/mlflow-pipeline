FROM python:3.9-slim

WORKDIR /app

# Install dependencies for MLflow and your model flavor
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install .

# Environment variables for MLflow connection
ENV MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
ENV MODEL_NAME=IrisClassifier
ENV MODEL_STAGE=Production

COPY serve.py .

# Expose port for the API
EXPOSE 8080

# Healthcheck: used by the deployment pipeline to verify readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "serve:app"]
