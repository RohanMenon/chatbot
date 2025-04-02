import kagglehub
import os
import json

def _set_credentials_json() -> None:
    """Set credentials from a JSON file."""
    try:
        with open("kaggle.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            os.environ["KAGGLE_USERNAME"] = data["username"]
            os.environ["KAGGLE_KEY"] = data["key"]
    except FileNotFoundError as e:
        print(f"Error loading Kaggle credentials: {e}")

def download_model() -> str:
    """Download and return the path to the model files."""
    _set_credentials_json()
    # Download latest version
    path = kagglehub.model_download("google/gemma-2/transformers/gemma-2-2b-it")

    print("Path to model files:", path)
    return path
