import logging
import sys
import os
import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import ks_2samp
from typing import Optional
from ShipmentPricePrediction.exception import ShipmentPriceException
from ShipmentPricePrediction.entity import config_entity
from ShipmentPricePrediction.entity import artifact_entity
from ShipmentPricePrediction.config import TARGET_COLUMN
from ShipmentPricePrediction import utils


class DataValidation:
    def __init__(self, data_validation_config: config_entity.DataValidationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"********Data Validation********")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()

        except Exception as e:
            raise ShipmentPriceException(e, sys)

    def drop_missing_values_columns(self, df: pd.DataFrame, report_key_name: str) -> [pd.DataFrame]:
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum() / df.shape[0]
            drop_columns_names = null_report[null_report > threshold].index
            self.validation_error[report_key_name] = list(drop_columns_names)
            logging.info(f"threshold: {threshold}, null_report:{null_report}, drop_columns_names: {drop_columns_names}")

            df.drop(list(drop_columns_names), axis=1, inplace=True)

        except Exception as e:
            raise ShipmentPriceException(e, sys)

    def is_required_column_exists(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> bool:
        try:
            base_columns = base_df
            current_columns = current_df
            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"columns : {base_column} is not available in current column: {current_columns}")
                    missing_columns.append(base_column)

            if len(missing_columns) > 0:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True

        except Exception as e:
            raise ShipmentPriceException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str):
        try:
            drift_report = dict()
            base_columns = base_df.columns
            current_columns = current_df.columns
            for base_column in base_columns:
                base_data, current_data = base_df[base_columns], current_df[current_columns]
                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    # Null Hypothesis accept
                    drift_report[base_column] = {
                        "p_values": float(same_distribution.pvalue),
                        "same_distribution": True
                    }

                else:
                    drift_report[base_column] = {
                        "p_values": float(same_distribution.pvalue),
                        "same_distribution": False
                    }
            self.validation_error[report_key_name] = drift_report

        except Exception as e:
            raise ShipmentPriceException(e, sys)

    def initiate_data_validation(self):
        try:
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            # base_df.replace({"na": np.NAN}, inplace=True)
            # base_df = self.drop_missing_values_columns(df=base_df, report_key_name="missing_value_within_base_dataset")

            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            # logging.info(f"train_df :{train_df.shape} and test_df: {test_df.shape}")
            # logging.info(f"train_file_path :{self.data_ingestion_artifact.train_file_path} and test_file_path :{self.data_ingestion_artifact.test_file_path}")

            # train_df = self.drop_missing_values_columns(df=train_df,report_key_name="missing_values_within_base_dataset")
            #test_df = self.drop_missing_values_columns(df=test_df, report_key_name="missing_values_within_base_dataset")

            exclude_columns = [TARGET_COLUMN]
            logging.info(f"base_df.columns {base_df.columns}")
            base_df = utils.convert_columns_float(base_df, exclude_columns=exclude_columns)
            logging.info(f"base_df convert columns float done")
            train_df = utils.convert_columns_float(train_df, exclude_columns=exclude_columns)
            logging.info(f"train_df convert columns float done")
            test_df = utils.convert_columns_float(test_df, exclude_columns=exclude_columns)
            logging.info(f"test_dfconvert columns float done")
            """train_df_columns_status = self.is_required_column_exists(base_df=base_df, current_df=train_df,
                                                                     report_key_name='missing_columns_within_train_dataset')
            test_df_columns_status = self.is_required_column_exists(base_df=base_df, current_df=test_df,
                                                                    report_key_name='missing_columns_within_test_dataset')

            if train_df_columns_status:
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_within_test_dataset")

            if test_df_columns_status:
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_within_test_dataset")"""

            logging.info(f"this is check in data validation")
            logging.info(f"file_path=self.data_validation_config.report_file_path{self.data_validation_config.report_file_path} and data=self.validation_error{self.validation_error}")

            # Write your yaml report
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path, data=self.validation_error)
            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            return data_validation_artifact

        except Exception as e:
            raise ShipmentPriceException(e, sys)