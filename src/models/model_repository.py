import mlflow
import mlflow.sklearn

class ModelRepository:

    def __init__(self, experiment_name="Bank_Marketing_Models", tracking_uri="file:../src/models/mlruns"):
        
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def log_model(self, model, model_name, params, metrics):

        with mlflow.start_run() as run:

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, model_name)
            run_id = run.info.run_id

            return run_id

    def register_model(self, run_id, model_name, registered_name):

        model_uri = f"runs:/{run_id}/{model_name}"
        try:
            result = mlflow.register_model(model_uri, registered_name)
            print(f"Model registered: {result.name}")
            return result
        except Exception as e:
            print("Model registration failed:", e)
            return None

    def load_model(self, registered_name, stage="None"):

        model_uri = f"models:/{registered_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        return model

    def list_registered_models(self):

        client = mlflow.tracking.MlflowClient()
        return client.list_registered_models()