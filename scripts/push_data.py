import os, sys, json
from dotenv import load_dotenv
import certifi
import pandas as pd
import numpy as np
from src.exception.exception import CustomException
from src.logging.logger import get_logger

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


load_dotenv()
uri = os.getenv("MONGO_DB_URL")
database = os.getenv("DATABSE")

data_file_path = os.getenv("DATA_FILE_PATH")
data_collection = os.getenv("DATA_COLLECTION_NAME")

batch_file_path = os.getenv("BATCH_FILE_PATH")
batch_collection = os.getenv("BATCH_COLLECTION_NAME")

ca = certifi.where()

logging = get_logger(__name__)

class RestaurantDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def csv_to_json_converter(self, file_path:str) -> json:
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def insert_data_mongodb(self, records:json, database:str, collection:str):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=ca)

            self.database=self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)

            return (len(self.records))

        except Exception as e:
            raise CustomException(e, sys) from e
        
if __name__=="__main__":
    try:
        restaurant_obj=RestaurantDataExtract()

        # Exporting main data file to MongoDB
        data_records=restaurant_obj.csv_to_json_converter(file_path=data_file_path)
        logging.info("Data converted to json format")
        no_of_data_records=restaurant_obj.insert_data_mongodb(records=data_records, database=database, collection=data_collection)
        logging.info(f"{no_of_data_records} Records pushed into Database")

        # Exporting Test data file to MongoDB
        batch_records=restaurant_obj.csv_to_json_converter(file_path=batch_file_path)
        logging.info("Data converted to json format")
        no_of_batch_records=restaurant_obj.insert_data_mongodb(records=batch_records, database=database, collection=batch_collection)
        logging.info(f"{no_of_batch_records} Records pushed into Database")
    except:
        print("Operation Failure!!")