import logging
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional
from ShipmentPricePrediction.config import TARGET_COLUMN
from ShipmentPricePrediction import utils
from ShipmentPricePrediction.logger import logging
from ShipmentPricePrediction.entity import config_entity
from ShipmentPricePrediction.entity import artifact_entity
from ShipmentPricePrediction.exception import ShipmentPriceException
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from ShipmentPricePrediction.predictor import ModelResolver
from ShipmentPricePrediction.utils import load_object



class ModelEvaluation:
    def __init__(self, model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainerArtifact):

        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise ShipmentPriceException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_accuracy=None)
                logging.info(f"model evaluation artifact: {model_eval_artifact}")
                logging.info(f"Model Evaluation Phase Done.")
                return model_eval_artifact
            # Find location previous Model

            transformers_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            # target_encoder_path  = self.model_resolver.get_latest_target_encoder_path()

            # Previous Model
            transformers = load_object(file_path=transformers_path)
            model = load_object(file_path=model_path)
            # target_encoder = load_object(file_path=target_encoder_path)
            logging.info(f"transformers: {transformers} and model: {model}")

            # Current Model
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            # current _target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df =test_df[TARGET_COLUMN]

            y_true = target_df

            input_features_name = list(transformers.feature_names_in_)

            """for i in input_features_name:
                if test_df[i].dtypes == 'object':
                    test_df[i]=target_encoder.fit_transform(test_df[i])"""
            input_arr = transformers.transform(test_df[input_features_name])
            y_pred = model.predict(input_arr)

            # Model Comparsion b/w New  and Old Model.
            previous_model_score = r2_score(y_true, y_pred)

            # Accuracy Current Model
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr = current_transformer.transform(test_df[input_feature_name])
            y_pred = current_model.predict(input_arr)
            y_true = target_df
            # current_target_encoder.transform(target_df)
            # current_target_encoder.inverse_transform(y_pred[:5])
            print(f"Prediction using trained model: {y_pred[:5]}")
            current_model_score = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model: {current_model_score}")
            if current_model_score <= previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                # raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                          improved_accuracy=current_model_score - previous_model_score)
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            logging.info(f"Model Eval Completed.")
            return model_eval_artifact

        except Exception as e:
            raise ShipmentPriceException(e, sys)


