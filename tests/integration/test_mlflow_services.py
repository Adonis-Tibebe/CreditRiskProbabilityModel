import os
import tempfile
import shutil
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
import mlflow
from src.services.mlflow_services import log_model_run

def test_log_model_run_outside_project(monkeypatch):
    # Create a temp directory outside the project root for MLflow artifacts
    temp_dir = tempfile.mkdtemp(dir=os.path.expanduser("~"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", temp_dir)

    # Minimal model and data
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y = [0, 1, 0, 1]
    model = RandomForestClassifier(n_estimators=1, random_state=42)
    model.fit(X, y)

    params = {"n_estimators": 1}
    metrics = {"accuracy": 1.0}

    # Skip test if registry is not supported
    tracking_uri = mlflow.get_tracking_uri()
    if os.path.abspath(tracking_uri) == os.path.abspath(temp_dir):
        pytest.skip("MLflow Model Registry not supported with local file store.")

    # Should not raise
    log_model_run(
        model_name="TestModel",
        params=params,
        metrics=metrics,
        model=model,
        X_sample=X,
        y_sample=y,
        artifact_path="test_model"
    )

    # Clean up temp directory after test
    shutil.rmtree(temp_dir)