import logging
import os
import sys
import numpy as np
import pandas as pd
from ShipmentPricePrediction.config import TARGET_COLUMN
from ShipmentPricePrediction import utils
from ShipmentPricePrediction.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from ShipmentPricePrediction.exception import ShipmentPriceException
from ShipmentPricePrediction.entity import config_entity, artifact_entity
from ShipmentPricePrediction.utils import get_numerical_and_categorical_columns


# Missing values impute
# Outlier's Handling
# Imbalance data Handling
# Covert Categorical data into numerical data


class DataTransformation:

    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise ShipmentPriceException(e, sys)

    """def get_num_and_cat_col(self):
        num_col, cat_col = get_numerical_and_categorical_columns(self.data_ingestion_artifact.feature_store_file_path)
        logging.info(f"num_col: {num_col} and cat_col: {cat_col}")
        return num_col, cat_col"""

    def get_data_transformer_object(self):
        try:
            numerical_columns, categorical_columns = utils.get_numerical_and_categorical_columns(
                self.data_ingestion_artifact.feature_store_file_path)
            logging.info(f"Getting num and cat columns in get_data_transformer_object function line no 45 :{numerical_columns} and  {categorical_columns}")

            """numerical_columns = ['Unit_of_Measure_Per_Pack', 'Line_Item_Quantity', 'Line_Item_Value',
                                 'Pack_Price', 'Unit_Price', 'Weight_Kilograms',
                                 'Line_Item_Insurance_USD']
            categorical_columns = ['Country', 'Managed_By', 'Fulfill_Via', 'Shipment_Mode',
                                   'Product_Group', 'Sub_Classification',
                                   'Brand',
                                   'First_Line_Designation']  # 'Item_Description'  , 'Vendor','Dosage', 'Dosage_Form' 
                                   """

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse=False, handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))])

            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, numerical_columns),
                 ("cat_pipelines", cat_pipeline, categorical_columns)])
            logging.info(f"Preprocessor Object : {preprocessor}")
            return preprocessor

        except Exception as e:
            raise ShipmentPriceException(e, sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"train_df : initiate_data_transformation : {train_df}")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"test_df : initiate_data_transformation : {test_df}")

            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            logging.info(f"input_feature_train_df: initiate_data_transformation : {input_feature_train_df}")
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)
            logging.info(f"input_feature_test_df: initiate_data_transformation : {input_feature_test_df}")

            # selecting target feature for train and test data frame
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info(f"target_feature_train_df:{target_feature_train_df} and target_feature_test_df:{target_feature_test_df}")

            # target_feature_train_arr = target_feature_train_df.squeeze()
            # target_feature_test_arr = target_feature_test_df.squeeze()

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            preprocessing_obj = self.get_data_transformer_object()
            logging.info(f"preprocessing_obj : {preprocessing_obj}")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # target encoder
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"train_arr by doing operation np.c_ {train_arr} and test_arr by doing operation np.c_{test_arr}")

            # Saving train array and test array

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            logging.info(f"Saving preprocessing object")

            utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=preprocessing_obj)
            # utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)
            # utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=preprocessing_obj)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path)
            return data_transformation_artifact

        except Exception as e:
            raise ShipmentPriceException(e, sys)


# target_encoder_path = self.data_transformation_config.target_encoder_path



