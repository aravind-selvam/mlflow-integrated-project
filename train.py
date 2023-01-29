"""Training module"""

import os
import time
import shutil
import mlflow
import pickle
import pandas as pd
import xgboost as xgb
from prefect import flow, task
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin
from sklearn.svm import SVR
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from six.moves import urllib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from prefect.task_runners import SequentialTaskRunner
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "used-car-prediction")
MODEL_SEARCH_ITERATIONS = int(os.getenv("MODEL_SEARCH_ITERATIONS", "1"))


@task
def download_data():
    """
    It downloads the data from the specified url, creates a directory for the dataset, renames the data
    with the current datetime, and sets the working data
    
    Returns:
      The data file path
    """
    download_url = 'https://raw.githubusercontent.com/aravind9722/datasets-for-ML-projects/main/cardekho_dataset.csv'

    download_path = 'data'

    # make directory for dataset
    os.makedirs(download_path, exist_ok=True)

    data_file_name = os.path.basename(download_url)

    data_file_path = os.path.join(download_path, data_file_name)

    urllib.request.urlretrieve(download_url, data_file_path)

    datetime = time.strftime("%Y%m%d-%H%M%S")
    # Rename data with current datetime
    os.rename(
        f"{download_path}/cardekho_dataset.csv",
        f"{download_path}/data-{datetime}.csv",
    )
    # Set working data
    shutil.copy(
        f"{download_path}/data-{datetime}.csv",
        f"{download_path}/data.csv",
    )
    data = f"{download_path}/data.csv"

    print(data)
    return data


@task
def read_data(filename):
    """
    It reads the data from the CSV file, drops the car_name and Unnamed: 0 columns, and then creates a
    pickle file with the model_list and brand_list.
    
    Args:
      filename: The name of the CSV file
    
    Returns:
      The dataframe
    """
    df = pd.read_csv(filename)

    df.drop(['car_name', 'Unnamed: 0'], axis=1, inplace=True)

    model_list = df["model"].unique()
    brand_list = df["brand"].unique()

    # make directory for pickle files
    os.makedirs("pickle", exist_ok=True)

    file = open("pickle/carnamelist.pkl", "wb")
    pickle.dump(model_list, file)
    pickle.dump(brand_list, file)
    file.close()

    return df


@task
def prepare_data(df):
    """
    It takes a dataframe, drops the target column, and then creates a preprocessor object that will be
    used to transform the dataframe. 
    
    The preprocessor object is then saved to a pickle file and logged as an artifact. 
    
    The preprocessor object is then used to transform the dataframe and the transformed dataframe is
    returned. 
    
    Args:
      df: The dataframe that contains the data we want to use for training and testing.
    
    Returns:
      X and y
    """
    target = 'selling_price'

    X = df.drop(target, axis=1)
    y = df[target]

    # Create Column Transformer with 3 types of transformers
    num_features = X.select_dtypes(exclude="object").columns
    onehot_columns = ['seller_type', 'fuel_type', 'transmission_type']
    binary_columns = ['brand', 'model']

    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()
    binary_transformer = BinaryEncoder()

    with mlflow.start_run():

        preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, onehot_columns),
                ("StandardScaler", numeric_transformer, num_features),
                ("BinaryEncoder", binary_transformer, binary_columns)

            ]
        )

        X = preprocessor.fit_transform(X)
        pickle.dump(preprocessor, open('pickle/preprocess.pkl', 'wb'))

    # Tracking our model
        mlflow.log_artifact(local_path="pickle/preprocess.pkl",
                            artifact_path="store_pickle")
    return X, y


@task
def split_data(X, y):
    """
    Splits data in training and test datasets
    
    Args:
      X: The dataframe containing the features
      y: The target variable
    
    Returns:
      X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=1
    )

    return X_train, X_val, y_train, y_val


@task
def train_model_xgboost_search(X_train, X_val, y_train, y_val):
    """
    Searches for the best XGBoost prediction model
    """
    mlflow.xgboost.autolog()

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    def objective(params):
        """
        We're using the hyperopt library to define a search space for our XGBoost model. We're then using
        the hyperopt library to search through the search space and find the best hyperparameters for our
        model
                
        Returns:
          the best hyperparameters found by the hyperparameter search.
        """
        with mlflow.start_run():
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=10,
                evals=[(valid, 'validation')],
                early_stopping_rounds=5
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2 = r2_score(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:squarederror',
        'seed': 42
    }

    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=MODEL_SEARCH_ITERATIONS,
        trials=Trials(),
    )
    return


@task
def train_model_sklearn_search(X_train, X_val, y_train, y_val):
    """
    Searches for the best scikit-learn prediction model
    """
    mlflow.sklearn.autolog()

    def objective(params):

        with mlflow.start_run(run_name='SklearnModels'):
            regressor_type = params["type"]
            del params["type"]
            if regressor_type == "svm":
                reg = make_pipeline(
                    StandardScaler(), SVR(**params, verbose=False))
            elif regressor_type == "rf":
                reg = make_pipeline(
                    StandardScaler(), RandomForestRegressor(**params))
            elif regressor_type == "catboost":
                reg = make_pipeline(
                    StandardScaler(), CatBoostRegressor(**params))

            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2 = r2_score(y_val, y_pred)
            signature = infer_signature(X_val, reg.predict(X_val))
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

        return {"loss": rmse, "status": STATUS_OK}

    search_space = hp.choice(
        "regressor_type",
        [
            {
                "type": "svm",
                "C": hp.uniform("SVR_C", 0.5, 15),
                "gamma": hp.uniform("SVM_gamma", 0.05, 15),
                "kernel": hp.choice("kernel", ["linear", "poly","rbf"]),
            },
            {
                "type": "rf",
                "max_depth": scope.int(hp.uniform("max_depth", 2, 5)),
                "criterion": hp.choice("criterion", ["squared_error", "absolute_error", "poisson"]),
            },
            {   "type": "catboost",
                'depth': hp.quniform("depth", 8, 16, 1),
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                'reg_lambda' : hp.uniform('reg_lambda', .1, 1),
                'subsample': hp.choice('subsample', [.7, .8, .9]),
                'n_estimators': hp.choice('n_estimators', [300,600])
            }
        ],
    )

    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=MODEL_SEARCH_ITERATIONS,
        trials=Trials(),
    )


@task
def register_best_model():
    """
    Registers the highest accuracy model
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"],
    )[0]
    # register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    best_rmse = round(best_run.data.metrics['rmse'])
    best_r2 = best_run.data.metrics['r2_score']
    model_details = mlflow.register_model(
        model_uri=model_uri, name=EXPERIMENT_NAME)
    client.update_registered_model(
        name=model_details.name, description=f"Current rmse: {best_rmse} \n best r2_score: {best_r2}"
    )


@flow(task_runner=SequentialTaskRunner())
def main():
    """
    It downloads the data, reads it, prepares it, splits it, trains two models, and registers the best
    one
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    path = download_data()
    data = read_data(path).result()
    X, y = prepare_data(data).result()
    X_train, X_val, y_train, y_val = split_data(X, y).result()
    train_model_xgboost_search(X_train, X_val, y_train, y_val)
    train_model_sklearn_search(X_train, X_val, y_train, y_val)
    register_best_model()


if __name__ == "__main__":
    main()
