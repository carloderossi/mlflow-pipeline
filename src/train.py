from unittest import result
from xml.parsers.expat import model
from xmlrpc import client

import mlflow
from mlflow import data
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature

from mlflow.tracking import MlflowClient

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import os

def log_multiclass_roc_curve(model, X_test, y_test):
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_proba = model.predict_proba(X_test)

    plt.figure()
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve (OvR)")
    plt.legend(loc="lower right")

    os.makedirs("plots", exist_ok=True)
    path = "plots/multiclass_roc_curve.png"
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path)

def add_description(result):
    client = MlflowClient()
    model_uri = result.model_uri  # e.g. "models:/IrisClassifier/2"
    name, version = model_uri.split("/")[-2:]

    client.update_model_version(
        name=name,
        version=version,
        description="CDR - A RandomForestClassifier trained on the Iris dataset with schema validation."
    )

def train_and_register():
    
    # 1. Setup - Load as a Pandas DataFrame
    print("Loading Iris dataset...")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    print(" Train Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mlflow.set_experiment("CDR-IrisClassifierExperiment")

    mlflow.autolog()
    print("Starting MLflow run...")
    with mlflow.start_run():
        # 2. Train
        print("Training RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # 3. Log
        print("Logging model and metrics to MLflow...")
        mlflow.log_metric("train_accuracy", model.score(X_train, y_train))
        mlflow.log_metric("test_accuracy", model.score(X_test, y_test))
        mlflow.log_metric("accuracy", model.score(X_test, y_test))

        from sklearn.metrics import confusion_matrix
        import pandas as pd
        cm = confusion_matrix(y_test, model.predict(X_test))
        pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)
        mlflow.log_artifact("confusion_matrix.csv")
        
        # 4. Register to Model Registry
        # This creates a new version under the name "IrisClassifier"
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path="model",
        #     registered_model_name="IrisClassifier"
        # )
        print("Generating model signature and registering model...")
        # Generate a signature by passing sample input and its prediction
        # A Model Signature is a contract. It defines the required inputs (columns and types) and the expected output. 
        # In production, this prevents your model from trying to predict on malformed data.
        signature = infer_signature(X_test, model.predict(X_test))

        # Log the model with the signature
        print("Logging model with signature to MLflow...")
        result = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="CDRRandomForestClassifiermodel_with_signature", #artifact_path="model",
            signature=signature,           # Enforces schema on load
            input_example=X_test.iloc[:3], # Adds a UI-visible example
            registered_model_name="IrisClassifier",
        )
        print(f"Model registered successfully: {result}")
        try:
            add_description(result)
        except Exception as e:
            print(f"Failed to add description: {e}")
        
        # 5. Save the training data as a CSV artifact
        # This becomes our 'Reference' for drift detection
        X_train.to_csv("train_reference.csv", index=False)
        mlflow.log_artifact("train_reference.csv")
        X_test.to_csv("test_reference.csv", index=False)
        mlflow.log_artifact("test_reference.csv")
        print("Training reference data logged to MLflow.")

        log_multiclass_roc_curve(model, X_test, y_test)

if __name__ == "__main__":
    train_and_register()