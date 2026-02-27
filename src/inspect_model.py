import mlflow
from mlflow.tracking import MlflowClient
import pprint

# 1. Configuration
MODEL_NAME = "IrisClassifier"
MODEL_STAGE = "Production" # Or use "1" for Version 1
# The '@' tells MLflow to look for an Alias instead of a version number. This is more robust for production use.
model_uri = f"models:/{MODEL_NAME}@Production"

def get_latest_model_version(client, model_name):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print(f"No versions found for model '{MODEL_NAME}'")
        return None

    # Filter for the latest version with no stage (equivalent to stage="None")
    none_stage_versions = [v for v in versions if v.current_stage == "None"]

    # Sort by version number and take the latest
    latest_version = max(none_stage_versions, key=lambda v: int(v.version))

    print("Latest version:", latest_version.version)

def dump_model_info():
    client = MlflowClient()

    # Find the latest version
    print(f"Finding latest version of model '{MODEL_NAME}'...")
    latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
    print(f"Latest version found: {latest_version}")
    try:
        get_latest_model_version(client, MODEL_NAME)
    except Exception as e:
        print(f"Error retrieving latest model version: {e}")

    # Assign the 'Production' alias to it
    print(f"Assigning alias '{MODEL_STAGE}' to version {latest_version}...")
    client.set_registered_model_alias(MODEL_NAME, "Production", latest_version)

    # --- A. GET REGISTRY METADATA ---
    # This info comes from the Model Registry (MySQL/Postgres)
    print(f"\n=== [Registry Metadata: {model_uri}] ===")
    version_details = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE)
    
    pprint.pprint({
        "Version": version_details.version,
        "Aliases": version_details.aliases,
        "Creation Time": version_details.creation_timestamp,
        "Run ID": version_details.run_id,
        "Status": version_details.status,
        "Tags": version_details.tags,
        "Description": version_details.description
    })

    # --- B. GET MODEL FILE METADATA ---
    # This info comes from the 'MLmodel' file (S3/Local Folder)
    print("\n=== [Model File Metadata (MLmodel)] ===")
    model_info = mlflow.models.get_model_info(model_uri)
    
    print(f"Model URI: {model_info.model_uri}")
    print(f"Model Config (artifact_path): {model_info.artifact_path}")
    print(f"Flavors: {list(model_info.flavors.keys())}")
    
    # --- C. INSPECT THE SIGNATURE (The Contract) ---
    print("\n=== [Model Signature] ===")
    if model_info.signature:
        print("INPUTS:")
        pprint.pprint(model_info.signature.inputs.to_dict())
        print("\nOUTPUTS:")
        pprint.pprint(model_info.signature.outputs.to_dict())
    else:
        print("No signature found.")

    # --- D. LOAD THE ACTUAL OBJECT ---
    print("\n=== [Loaded Model Attributes] ===")
    model = mlflow.pyfunc.load_model(model_uri)
    # This is useful if you want to see specific internal params
    print(f"Model Object Type: {type(model)}")

    # --- E. RUN INFO ---
    run_id = model_info.run_id
    run = mlflow.get_run(run_id)
    print("\n=== [Run Info] ===")
    pprint.pprint({ 
        "Run ID": run.info.run_id,
        "Experiment ID": run.info.experiment_id,
        "Status": run.info.status,
        "Start Time": run.info.start_time,
        "End Time": run.info.end_time,
        "Metrics": run.data.metrics,
        "Params": run.data.params,
        "Tags": run.data.tags,
        "Data": run.data,
        
    })



if __name__ == "__main__":
    dump_model_info()
    # you can also run this command in the terminal to get the same info:
    # Pointing to your local database and artifact folder
    # mlflow ui --backend-store-uri sqlite:///mlflow.db
    # then open a brwser at http://localhost:5000 and navigate to the Model Registry section.
    # see also screenshot in Readme.md for what the UI looks like.