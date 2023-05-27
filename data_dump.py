from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri = "mongodb+srv://ritesh:ritesha17@cluster0.isnrpik.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)


DATA_FILE_PATH = "D:\ShipmentPricePredictionProject\SCMS_Delivery_History_Dataset.csv"
DATABASE_NAME = "SHIPMENTPRICE"
COLLECTION_NAME = "SHIPMENTPRICE_PROJECT"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns:{df.shape}")
    df.reset_index(drop=True, inplace=True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
