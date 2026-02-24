"""
Dynamic MLflow Inference Service
This script uses the mlflow.pyfunc flavor. 
This is a "production secret": even if you change your model from Scikit-Learn to XGBoost or PyTorch later, 
this script remains exactly the same because pyfunc provides a unified .predict() interface.
"""

import os
import mlflow.pyfunc
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Config: Pull these from environment variables set in Docker/K8s
MODEL_NAME = os.getenv("MODEL_NAME", "IrisClassifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the model globally at startup for low-latency inference
# URI format: 'models:/<name>/<stage>'
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
print(f"Loading {MODEL_STAGE} model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        
        # MLflow's Model Signature automatically validates 'df' schema here
        predictions = model.predict(df)
        
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Healthcheck endpoint for the CI/CD pipeline."""
    return jsonify({"status": "healthy", "model": model_uri}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)