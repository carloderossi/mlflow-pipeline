# 🚀 Production MLOps Pipeline

This repository demonstrates a complete, automated MLOps ecosystem using **MLflow**, **GitHub Actions**, and [**Evidently AI**](https://github.com/evidentlyai/evidently). It transitions a model from a training script to a governed, schema-enforced production API.

## 🏗️ System Architecture

1. **CI/CD (GitHub Actions):** Automates training and deployment.
2. **Experiment Tracking:** Uses MLflow to track parameters and isolate `Staging` vs `Production` runs.
3. **Model Registry:** Serves as the "Source of Truth" for model versions.
4. **Inference:** A Flask-based container that dynamically pulls models tagged as `Production`.
5. **Monitoring:** Drift detection dashboards via Evidently AI.

## 🛠️ Project Structure

```text
├── .github/workflows/
│   └── pipeline.yaml       # CI/CD automation logic
├── src/
│   ├── train.py            # Training + Registration with Signatures
│   └── monitor.py          # Drift detection logic
├── serve.py                # Production Inference API
├── Dockerfile              # Containerization with Healthchecks
└── requirements.txt
```

### Production Features

- **Model Signatures:** Every model is logged with a JSON schema (Input/Output).  
  If the API receives a string instead of a float, MLflow rejects the request before it hits the model.
- **Zero-Downtime Deployment:** The CI/CD pipeline uses Docker healthchecks to ensure the new model is loaded and ready before decommissioning the old one.
- **Environment Isolation:**  
  * *Staging:* `mlflow.set_experiment("/iris_staging")`  
  * *Production:* Managed via the Model Registry aliases.

## 🚀 Deployment Guide

1. **Local Setup**
   ```bash
   export MLFLOW_TRACKING_URI=http://your-server:5000
   pip install -r requirements.txt
   python src/train.py
   ```
2. **Running the Inference Service**
   ```bash
   docker build -t iris-service .
   docker run -p 8080:8080 \
     -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
     iris-service
   ```
3. **Triggering Drift Analysis**  
   To compare production performance against training data:
   ```bash
   python src/monitor.py --reference data/train.csv \
     --current data/production_logs.csv
   ```

Created as a demonstration of Production ML Systems Management.

## 💡 Summary of "Production" Touches

1. **The Signature:** In `train.py`, use `infer_signature(X_train, model.predict(X_train))`.
2. **The Healthcheck:** In `serve.py`, the `/health` endpoint is what the Docker engine uses to decide if the "Green" container is ready to take over from the "Blue" container.
3. **The Environment:** By setting `MODEL_STAGE=Production` in the Dockerfile, you can ensure the service only loads models labeled `Production`.

---

### 🏷️ Tags

`mlflow` `mlops` `model-registry` `Evidently AI` `pipelines` `monitoring`