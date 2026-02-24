import mlflow
import pandas as pd
from sklearn.datasets import load_iris

# 1. Configuration
MODEL_NAME = "IrisClassifier"
ALIAS = "Production"
model_uri = f"models:/{MODEL_NAME}@{ALIAS}"

def run_prediction():
    print(f"--- Loading Model: {model_uri} ---")
    
    # 2. Load the model as a PyFunc (Generic Python Function)
    # This is the most robust way to load models in production
    model = mlflow.pyfunc.load_model(model_uri)

    # 3. Prepare sample data
    # We use a DataFrame because our 'train.py' signature expects column names
    iris = load_iris()
    sample_data = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]], 
        columns=iris.feature_names
    )

    # 4. Predict
    prediction = model.predict(sample_data)
    
    # Map the numerical result back to the species name
    species = iris.target_names[prediction[0]]
    
    print(f"Input Data:\n{sample_data}")
    print(f"\nPrediction Result: {prediction[0]} ({species})")

if __name__ == "__main__":
    run_prediction()