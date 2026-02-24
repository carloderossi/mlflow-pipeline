import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from evidently.future.report import Report
from evidently.future.datasets import Dataset
from evidently.metric_preset import DataDriftPreset

# Import your training function
from src.train import train_and_register 

MODEL_NAME = "IrisClassifier"
ALIAS = "Production"

def run_monitoring():
    client = MlflowClient()
    
    # 1. Fetch Production Data
    v_details = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    local_path = client.download_artifacts(v_details.run_id, "train_reference.csv")
    reference_df = pd.read_csv(local_path)

    # 2. Simulate Production Data (With Drift)
    current_df = reference_df.copy()
    current_df['petal width (cm)'] = current_df['petal width (cm)'] * 5.0 

    # 3. Run Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    
    # 4. Check Drift Status Programmatically
    report_dict = drift_report.as_dict()
    # Evidently nests the results: metrics -> drift preset -> result -> dataset_drift
    drift_detected = report_dict['metrics'][0]['result']['dataset_drift']

    if drift_detected:
        print("\n⚠️ ALERT: Data Drift Detected!")
        print("Automatic Retraining Triggered...")
        # This calls your train.py logic to create a NEW version in MLflow
        train_and_register() 
        print("✅ Retraining complete. New model version registered.")
    else:
        print("\n✅ No drift detected. System healthy.")

    drift_report.save_html("drift_report.html")

if __name__ == "__main__":
    run_monitoring()