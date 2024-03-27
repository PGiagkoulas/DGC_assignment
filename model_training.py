import utils
import mlflow
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def __get_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = utils.read_data_file(utils.DATASET_FILE_LOCATION)
    return dataset.drop("price", axis=1), dataset["price"]


def train_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ('scl', Pipeline([('scale', StandardScaler())]), ["area"]),
            ('ohe', Pipeline([('ohe', OneHotEncoder())]), utils.CATEGORICAL_FEATURES)
        ],
        remainder='passthrough')
    model = RandomForestRegressor(
        # random_state=utils.RND_STATE,
        bootstrap=True)
    training_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    param_distributions = {
        'model__n_estimators': [int(j) for j in np.linspace(start=50, stop=500, num=5)],
        'model__criterion': ["squared_error", "friedman_mse"],
        'model__max_features': ['log2', 'sqrt'],
        'model__max_depth': [int(j) for j in np.linspace(5, 15, num=5)],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    rand_search = RandomizedSearchCV(
        training_pipeline
        , param_distributions
        , n_iter=100
        , cv=3
        , random_state=utils.RND_STATE
        , refit=True
        , verbose=2
    )

    x, y = __get_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=utils.RND_STATE)
    rand_search.fit(X=x_train, y=y_train)

    mlflow.set_tracking_uri(utils.MODEL_LOG_LOCATION)
    mlflow.set_experiment(utils.EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("PricePredictionModel").experiment_id):
        mlflow.log_params(rand_search.best_params_)
        mlflow.log_metric(rand_search.best_params_.get("model__criterion"), rand_search.best_score_)
        mlflow.log_metric("RandomSearch_test_score", rand_search.best_score_)
        mlflow.log_metric("RandomSearch_test_set_score", rand_search.best_estimator_.score(x_test, y_test))

        signature = mlflow.models.infer_signature(x_test, rand_search.best_estimator_.predict(x_test))

        mlflow.sklearn.log_model(
            sk_model=rand_search.best_estimator_,
            artifact_path="pricing_model",
            signature=signature,
            input_example=x_test,
            registered_model_name=utils.MODEL_NAME,

        )

        print(f"Test score after training: {rand_search.best_estimator_.score(x_test, y_test)}")

    mlc = mlflow.MlflowClient()
    latest_model_version = utils.get_latest_model_version(mlc, utils.MODEL_NAME)
    mlc.set_registered_model_alias(utils.MODEL_NAME, "latest_challenger", latest_model_version)

    utils.champion_model_transition(mlc, utils.MODEL_NAME)
