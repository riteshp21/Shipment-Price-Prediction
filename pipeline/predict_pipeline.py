import logging
import sys
import os
import pandas as pd
from ShipmentPricePrediction.exception import ShipmentPriceException
import pickle
from ShipmentPricePrediction.utils import load_object
from ShipmentPricePrediction.logger import logging
import os
import dill
from ShipmentPricePrediction.entity.artifact_entity import DataTransformationArtifact
from ShipmentPricePrediction.entity.artifact_entity import ModelTrainerArtifact

class PredictPipeline:
    def __init__(self):
        pass

    def predict_value(self, features):
        try:
            model_path = os.path.join(r"D:\ShipmentPricePredictionProject\artifact\05152023__200724\model_trainer\model\model.pkl")
            preprocessor_path = os.path.join(r"D:\ShipmentPricePredictionProject\artifact\05152023__200724\data_transformation\transformed\transformer.pkl")
            logging.info(f"model_path is {model_path}")
            logging.info(f"model_path is {preprocessor_path}")
            print("Before Loading")
            model = dill.load(open(model_path, "rb"))
            preprocessor = dill.load(open(preprocessor_path, "rb"))
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            ShipmentPriceException(e, sys)


class CustomData:
    def __init__(self,
                 Unit_of_Measure_Per_Pack: int,
                 Line_Item_Quantity: int,
                 Line_Item_Value: float,
                 Pack_Price: float,
                 Unit_Price: float,
                 Weight_Kilograms: int,
                 Line_Item_Insurance_USD: float,
                 Country: str,
                 Managed_By: str,
                 Fulfill_Via: str,
                 Shipment_Mode: str,
                 Product_Group: str,
                 Sub_Classification: str,
                 Brand: str,
                 First_Line_Designation: str):
        self.Unit_of_Measure_Per_Pack = Unit_of_Measure_Per_Pack
        self.Line_Item_Quantity = Line_Item_Quantity
        self.Line_Item_Value = Line_Item_Value
        self.Pack_Price = Pack_Price
        self.Unit_Price = Unit_Price
        self.Weight_Kilograms = Weight_Kilograms
        self.Line_Item_Insurance_USD = Line_Item_Insurance_USD
        self.Country = Country
        self.Managed_By = Managed_By
        self.Fulfill_Via = Fulfill_Via
        self.Shipment_Mode = Shipment_Mode
        self.Product_Group = Product_Group
        self.Sub_Classification = Sub_Classification
        self.Brand = Brand
        self.First_Line_Designation = First_Line_Designation

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Unit_of_Measure_Per_Pack": [self.Unit_of_Measure_Per_Pack],
                "Line_Item_Quantity": [self.Line_Item_Quantity], "Line_Item_Value": [self.Line_Item_Value],
                "Pack_Price": [self.Pack_Price], "Unit_Price": [self.Unit_Price],
                "Weight_Kilograms": [self.Weight_Kilograms],
                "Line_Item_Insurance_USD": [self.Line_Item_Insurance_USD], "Country": [self.Country],
                "Managed_By": [self.Managed_By],
                "Fulfill_Via": [self.Fulfill_Via], "Shipment_Mode": [self.Shipment_Mode],
                "Product_Group": [self.Product_Group],
                "Sub_Classification": [self.Sub_Classification], "Brand": [self.Brand],
                "First_Line_Designation": [self.First_Line_Designation]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise ShipmentPriceException(e, sys)


