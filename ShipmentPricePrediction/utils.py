import pandas as pd
import numpy as np
import os
import sys
import yaml
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from ShipmentPricePrediction.exception import ShipmentPriceException
from ShipmentPricePrediction.config import mongo_client
from ShipmentPricePrediction.logger import logging


def get_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    try:
        logging.info(f"Reading data from database :{database_name} and {collection_name}")
        df = pd.DataFrame(mongo_client[database_name][collection_name].find())
        # logging.info(f"Find Columns {df.columns} and Shape {df.shape} ")
        if '_id' in df.columns:
            logging.info(f"Dropping ID Columns")
            df = df.drop('_id', axis=1)

        logging.info(f"Rows and Columns: {df.shape}")
        return df

    except Exception as e:
        raise ShipmentPriceException(e, sys)


def convert_columns_float(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if df[column].dtypes != 'O':
                    df[column] = df[column].astype('float')
        return df
    except Exception as e:
        raise ShipmentPriceException(e, sys)


# file_path:D:\ShipmentPricePredictionProject\artifact\05012023__201310\data_validation\report.yaml
# file_dir:D:\ShipmentPricePredictionProject\artifact\05012023__201310\data_validation
# training_pipeline_config_artifact_dir  = r"D:\practice\artifact\05012023__092357\data_validation"
# data_validation_dir = os.path.join(training_pipeline_config_artifact_dir , "data_validation")
# report_file_path=os.path.join(data_validation_dir, "report.yaml") #file_path


def write_yaml_file(file_path, data: dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"file_path:{file_path}")
        logging.info(f"file_dir:{file_dir}")
        with open(file_path, "w") as file_write:  # replace file_dir to #file_path
            yaml.dump(data, file_write)

    except Exception as e:
        raise ShipmentPriceException(e, sys)


"""def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise(e, sys)
"""


def get_numerical_and_categorical_columns(path):
    df = pd.read_csv(path)
    numerical_columns = list(df.drop('Freight_Cost_USD', axis=1).select_dtypes(exclude="object").columns)
    categorical_columns = list(df.select_dtypes(include="object").columns)
    return numerical_columns, categorical_columns


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            logging.info("Exited the save_object method of utils")

            # pickle.dump(obj, file_obj)
            # dill.dump(obj, file_obj)

    except Exception as e:
        raise ShipmentPriceException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not available")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

        # return pickle.load(file_obj)
        # return dill.load(file_obj)

    except Exception as e:
        raise ShipmentPriceException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            logging.info(f"file_obj from utils function save_numpy_array_data{file_obj}")
            # logging.info(f"creating tar file nps")
            logging.info(f"type of file_obj: {type(file_obj)}")
            np.save(file_obj, array)

    except Exception as e:
        raise ShipmentPriceException(e, sys) from e

# Model Trainer


def load_numpy_array_data(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            logging.info(f"{file_obj}")
            return np.load(file_obj)

    except Exception as e:
        raise ShipmentPriceException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise ShipmentPriceException(e, sys)
