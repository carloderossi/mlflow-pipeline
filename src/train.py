import mlflow
from mlflow import data
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature

def train_and_register():
    # 1. Setup - Load as a Pandas DataFrame
    print("Loading Iris dataset...")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    print(" Train Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Starting MLflow run...")
    with mlflow.start_run():
        # 2. Train
        print("Training RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # 3. Log
        print("Logging model and metrics to MLflow...")
        mlflow.log_metric("accuracy", model.score(X_test, y_test))
        
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
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,           # Enforces schema on load
            input_example=X_test.iloc[:3], # Adds a UI-visible example
            registered_model_name="IrisClassifier"
        )
        print("Model registered successfully")

if __name__ == "__main__":
    train_and_register()