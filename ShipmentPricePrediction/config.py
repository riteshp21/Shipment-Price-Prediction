import pymongo
import pandas as pd
import numpy as np
import json
import os, sys
from dataclasses import dataclass


@dataclass
class EnvironmentVariable:
    mongo_db_url = os.getenv("MONGO_DB_URL")


env_var = EnvironmentVariable()
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
print(f"printing mongo client {mongo_client}")
TARGET_COLUMN = 'Freight_Cost_USD'
print(env_var.mongo_db_url)