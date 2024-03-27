import mlflow
import utils


def infer_on_model():
    # load model
    ml_client = mlflow.MlflowClient()
    loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{utils.MODEL_NAME}@champion")

    # run predictions on loaded model
    test_set = utils.read_data_file(utils.DATASET_FILE_LOCATION)
    test_score = loaded_model.score(test_set.drop('price', axis=1).head(), test_set['price'].head())
    print(f"Score on inference: {test_score}")
    return test_score
