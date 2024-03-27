import mlflow
import pandas as pd

BOOLEAN_FEATURES = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
CATEGORICAL_FEATURES = ["furnishingstatus"]
NUMERICAL_FEATURES = ["area", "bedrooms", "bathrooms", "stories"]
INPUT_FILE_LOCATION = "./housing_data.csv"
DATASET_FILE_LOCATION = "./dataset.csv"
MODEL_NAME = "rf_price_predictor"
MODEL_LOG_LOCATION = "./mlruns"
EXPERIMENT_NAME = "PricePredictionModel"

RND_STATE = 26032024


def read_data_file(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[!] File {file_path} was not found, raising exception!")
        raise


def store_data_file(data: pd.DataFrame, file_path: str) -> None:
    data.to_csv(file_path)


def get_all_model_versions(mlc: mlflow.MlflowClient, model_name: str):
    return mlc.search_model_versions(f"name='{model_name}'")


def get_latest_model_version(mlc: mlflow.MlflowClient, model_name: str) -> str:
    return get_all_model_versions(mlc, model_name)[0].version


def get_latest_model_version_run_id(mlc: mlflow.MlflowClient, model_name: str) -> str:
    return get_all_model_versions(mlc, model_name)[0].source


def champion_model_transition(mlc: mlflow.MlflowClient, model_name: str):
    # TODO: implement function - to check and update champion alias for best performing model
    #  currently only assigns champion to latest model and assigns last_champion to the previous version
    latest_model_version = get_latest_model_version(mlc, model_name)
    mlc.set_registered_model_alias(model_name, "champion", latest_model_version)
    if int(latest_model_version) > 1:
        mlc.set_registered_model_alias(model_name, "last_champion", str(int(latest_model_version)-1))
