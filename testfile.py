# from ShipmentPricePrediction.entity.artifact_entity import DataTransformationArtifact
from ShipmentPricePrediction.utils import load_object
from ShipmentPricePrediction.logger import logging
import os

model_path = os.path.join(r"D:\ShipmentPricePredictionProject\artifact\05142023__143127\model_trainer\model\model.pkl")
logging.info(f"model_path: {model_path}")
# preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
preprocessor_path = os.path.join(r"D:\ShipmentPricePredictionProject\artifact\05142023__143127\data_transformation\transformed\transformer.pkl")
logging.info(f"preprocessor_path: {preprocessor_path}")
model = load_object(model_path)
preprocessor = load_object(preprocessor_path)
p1 = preprocessor
logging.info(f"model: {model}")


