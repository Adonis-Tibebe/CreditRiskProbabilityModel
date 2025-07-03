import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def log_model_run(model_name, params, metrics, model, X_sample, y_sample, artifact_path="model"):
    """
    Logs parameters, metrics, model, and registers it in MLflow Model Registry.

    Parameters:
    - model_name (str): Name to register model under
    - params (dict): Hyperparameters used
    - metrics (dict): Performance metrics
    - model (sklearn estimator): Fitted model object
    - X_sample (DataFrame): Small sample of input features for signature inference
    - y_sample (Series or array): Corresponding sample labels
    - artifact_path (str): Subfolder to store model artifacts (default = "model")
    """
    mlflow.set_experiment("CreditRiskExperiment")  # Optional but helpful

    with mlflow.start_run() as run:
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Generate model signature from sample inputs
        signature = infer_signature(X_sample, model.predict(X_sample))

        # Log model with metadata
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            input_example=X_sample.iloc[:1],
            signature=signature
        )

        # Register the model in the MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
        mlflow.register_model(model_uri=model_uri, name=model_name)