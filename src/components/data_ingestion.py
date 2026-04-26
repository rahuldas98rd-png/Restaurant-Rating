import os, sys
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split

from src.exception.exception import CustomException
from src.logging.logger import get_logger

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.main_utils.utils import save_csv, read_yaml_file, fetch_data_from_database
from src.constants.training_pipeline import SCHEMA_FILE_PATH

import streamlit as st

load_dotenv()

def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

mongodb_url = _get_secret("MONGO_DB_URL")

ca = certifi.where()

logging = get_logger(__name__)


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def import_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name

            df = fetch_data_from_database(database_name=database_name,collection_name=collection_name)
            return df

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            save_csv(file_path=feature_store_file_path, dataframe=dataframe)
            logging.info(f"Dataframe saved in -> {feature_store_file_path}")
            return dataframe
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            test_size = self.data_ingestion_config.train_test_split_ratio
            random = self.data_ingestion_config.random_seed

            target_column = self._schema_config['data']['target_column']
            labels = self._schema_config['data']['rating_labels']

            rating_bucket = pd.cut(
                dataframe[target_column],
                bins=[-float("inf"), 2.5, 3.5, 4.0, float("inf")],
                labels=labels
            )

            logging.info("Initiate train test split on the dataframe with"\
                        "test_size: {} and random state: {}".format(test_size, random))
            train_set, test_set = train_test_split(dataframe,
                                                   test_size=test_size,
                                                   random_state=random,
                                                   stratify=rating_bucket)
            logging.info("Train test split successful")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train data to -> {self.data_ingestion_config.training_file_path}")
            save_csv(file_path=self.data_ingestion_config.training_file_path, dataframe=train_set)

            logging.info(f"Exporting test data to -> {self.data_ingestion_config.testing_file_path}")
            save_csv(file_path=self.data_ingestion_config.testing_file_path, dataframe=test_set)
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Start data ingestion process....")
            dataframe=self.import_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe=dataframe)
            self.split_data_as_train_test(dataframe=dataframe)

            logging.info("Initializing Data Ingestion Artifact")
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                            test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"Sucessfully initialized DataIngestionArtifact: {data_ingestion_artifact}")
            logging.info("Data Ingestion completed successfully.")
            logging.info("="*60)

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e