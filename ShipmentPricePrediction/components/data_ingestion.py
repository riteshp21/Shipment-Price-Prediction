from ShipmentPricePrediction.logger import logging
import pandas as pd
import numpy as np
import os
import sys
from ShipmentPricePrediction.exception import ShipmentPriceException
from ShipmentPricePrediction.entity import config_entity
from ShipmentPricePrediction.entity import artifact_entity
from ShipmentPricePrediction import utils
from sklearn.model_selection import train_test_split


class DataIngestion:  # data divided in train,test and validate
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ShipmentPriceException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"export collection data as pandas dataframe")
            df: pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"save data in feature store")

            # Replace na with NaN
            # df.replace(to_replace="na", value=np.NAN, inplace=True)

            # df = pd.read_csv('D:\IntershipProject\SCMS_Delivery_History_Dataset.csv')

            logging.info("Read the dataset from dataframe")
            # logging.info(f"No of columns{len(df.columns)} ")

            index1 = df[df["Freight Cost (USD)"].str.contains("Freight|Invoiced|See", case=False)].index
            index2 = df[df["Weight (Kilograms)"].str.contains("See|Weight")].index
            index3 = list(set(index1).union(set(index2)))
            df.drop(index3,inplace=True)
            df["Freight Cost (USD)"] = pd.to_numeric(df["Freight Cost (USD)"])
            df["Weight (Kilograms)"] = pd.to_numeric(df["Weight (Kilograms)"])
            df["Line Item Insurance (USD)"] = df["Line Item Insurance (USD)"].fillna(df["Line Item Insurance (USD)"].median())
            df.drop(['ID', 'Project Code', 'ASN/DN #', 'Delivered to Client Date', 'Delivery Recorded Date',
                          'Scheduled Delivery Date', 'Vendor INCO Term', 'PQ First Sent to Client Date',
                          'PO / SO #', 'PQ #', 'PO Sent to Vendor Date', 'Manufacturing Site',
                          'Molecule/Test Type', 'Item Description', 'Vendor', 'Dosage', 'Dosage Form'], axis=1, inplace=True)
            df["Shipment Mode"] = df["Shipment Mode"].fillna(df["Shipment Mode"].mode()[0])
            # df["Dosage"] = df["Dosage"].fillna(df["Dosage"].mode()[0])
            df.columns = df.columns.str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
            # logging.info(f"No of columns{len(df.columns)}")

            numerical_columns = df.select_dtypes(exclude="object").columns
            categorical_columns = df.select_dtypes(include="object").columns
            # logging.info(f"numerical_columns : {numerical_columns} and categorical_columns: {categorical_columns}")
            # logging.info(f"Our added code worked no error found {df.shape}")

            #**********************************************************************************************************************************************
            # Save data in feature store folder

            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            logging.info("Save df to feature store folder")
            # Save dataframe to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False, header=True)

            logging.info("Splitting data in train and test")
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=1)

            logging.info("Create dataset folder if not exist")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)
            logging.info(f"saving dataset to feature store folder")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            # Prepare Artifact folder
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
                )
            logging.info(f"Artifact folder Preparation Done in Data Ingestion.")
            return data_ingestion_artifact

        except Exception as e:
            raise ShipmentPriceException(error_message=e, error_detail=sys)



