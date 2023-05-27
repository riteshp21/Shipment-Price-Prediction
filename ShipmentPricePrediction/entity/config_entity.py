import os, sys
from ShipmentPricePrediction.exception import ShipmentPriceException
from datetime import datetime
from ShipmentPricePrediction.logger import logging


FILE_NAME = "shipmentprice.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE = "transformer.pkl"
MODEL_FILE_NAME = "model.pkl"


class TrainingPiplineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifact", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise ShipmentPriceException(e, sys)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPiplineConfig):
        try:
            self.database_name = "SHIPMENTPRICE"
            self.collection_name = "SHIPMENTPRICE_PROJECT"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_size = 0.2

        except Exception as e:
            raise ShipmentPriceException(e, sys)

    # convert data into dict
    def to_dict(self) -> dict:
        try:
            return self.__dict__
        except Exception as e:
            raise ShipmentPriceException(e, sys)


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPiplineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")  # yaml, json, csv
        self.missing_threshold: float = 0.2
        self.base_file_path = os.path.join("SCMS_Delivery_History_Dataset.csv")


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPiplineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")

        self.transform_object_path = os.path.join(self.data_transformation_dir, "transformed", TRANSFORMER_OBJECT_FILE)

        # This line run without error but need to add code after hash in run code again as the file generated in
        # data_transformation seems corrupted.

        self.transformed_train_path = os.path.join(self.data_transformation_dir, "transformed",
                                                   TRAIN_FILE_NAME)#.replace("csv", "npz")

        self.transformed_test_path = os.path.join(self.data_transformation_dir, "transformed",
                                                  TEST_FILE_NAME)#.replace("csv", "npz")


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPiplineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir, "model", MODEL_FILE_NAME)
        self.expected_accuracy = 0.5


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPiplineConfig):
        self.change_threshold = 0.8


class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPiplineConfig):
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir,"model_pusher_file")
        self.saved_model_dir = os.path.join("saved_models")
        self.pusher_model_dir = os.path.join(self.model_pusher_dir, "saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir, MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir, TRANSFORMER_OBJECT_FILE)
        # self.pusher_target_encoder_path = os.path.join(self.pusher_model_dir, TARGET_ENCODER_OBJECT_FILE_NAME)



