import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature

def train_and_register():
    # 1. Setup
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

    with mlflow.start_run():
        # 2. Train
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # 3. Log
        mlflow.log_metric("accuracy", model.score(X_test, y_test))
        
        # 4. Register to Model Registry
        # This creates a new version under the name "IrisClassifier"
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path="model",
        #     registered_model_name="IrisClassifier"
        # )
        # Generate a signature by passing sample input and its prediction
        # A Model Signature is a contract. It defines the required inputs (columns and types) and the expected output. 
        # In production, this prevents your model from trying to predict on malformed data.
        signature = infer_signature(X_test, model.predict(X_test))

        # Log the model with the signature
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,           # Enforces schema on load
            input_example=X_test.iloc[:3], # Adds a UI-visible example
            registered_model_name="IrisClassifier"
        )

if __name__ == "__main__":
    train_and_register()