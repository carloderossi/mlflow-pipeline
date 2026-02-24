"""
Drift Detection with Evidently AI: ML Model Monitoring with Evidently 
This script demonstrates how to use the Evidently library to monitor data drift in a production ML model.
In a real-world scenario, you would automate this script to run at regular intervals (e.g., daily) 
and alert ops/eng teams if significant drift is detected.
See also Agentic self-healing pipeline that can automatically trigger retraining or rollback based on drift detection."""
from library.evidently.report import Report
from library.evidently.metric_preset import DataDriftPreset
import pandas as pd

# 1. Load your 'Reference' data (Training set)
reference_data = pd.read_csv("train_baseline.csv")

# 2. Load 'Current' data (Live production logs)
current_data = pd.read_csv("production_logs.csv")

# 3. Generate Report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_data, current_data=current_data)

# 4. Save as HTML (to be hosted or sent as an artifact)
drift_report.save_html("drift_report.html")