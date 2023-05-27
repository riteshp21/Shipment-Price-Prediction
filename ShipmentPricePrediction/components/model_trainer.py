import logging
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional
from ShipmentPricePrediction.config import TARGET_COLUMN
from ShipmentPricePrediction import utils
from ShipmentPricePrediction.logger import logging
from ShipmentPricePrediction.entity import config_entity, artifact_entity
from ShipmentPricePrediction.exception import ShipmentPriceException
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# Model Trainer

class ModelTrainer:
    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):

        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ShipmentPriceException(e, sys)

    def train_model(self, x, y):
        try:
            pass
        except Exception as e:
            raise ShipmentPriceException(e, sys)

    def initiate_model_trainer(self) -> artifact_entity.ModelTrainerArtifact:
        try:
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            logging.info(f"train_arr from model_trainer initiate_model_trainer line no 47 is {train_arr}")
            logging.info(f"test_arr from model_trainer initiate_model_trainer line no 48 is{test_arr}")

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            # model = self.train_model(x=x_train, y=y_train)


            """ models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }"""
            models = {
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
            }

            """params ={"Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }}
                   """
            params = {
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }}

            model_report: dict = utils.evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                       models=models, param=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            """if best_model_score < self.model_trainer_config.expected_accuracy:
                raise ShipmentPriceException()"""

            logging.info(f"Best found model on both training and testing dataset")

            utils.save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model
            )
            x_test_prediction = best_model.predict(x_test)
            x_train_prediction = best_model.predict(x_train)
            r2_test_square = r2_score(y_test, x_test_prediction)
            r2_train_square = r2_score(y_train, x_train_prediction)
            logging.info(f"Best model is :{best_model}, r2_score is {r2_test_square}")
            print(f"Best model is :{best_model}, r2_score is {r2_test_square}")

            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                r2_train_score=r2_train_square, r2_test_score=r2_test_square)
            logging.info(f"Model Trainer Done")
            return model_trainer_artifact

        except Exception as e:
            raise ShipmentPriceException(e, sys)



