import os, sys, yaml, pickle, json
from src.exception.exception import CustomException
from src.logging.logger import get_logger
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi

import streamlit as st

load_dotenv()

def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

mongodb_url = _get_secret("MONGO_DB_URL")
database    = _get_secret("DATABSE")
ca = certifi.where()

logging = get_logger(__name__)


def save_csv(file_path: str, dataframe: pd.DataFrame, replace: bool = False) -> None:
    try:
        logging.info(f"Storing *.csv file to -> {file_path}")
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataframe.to_csv(file_path, index=False, header=True)
    except Exception as e:
        raise CustomException(e, sys) from e
    

def read_csv(file_path: str) -> pd.DataFrame:
    try:
        logging.info(f"Reading *.csv file from {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e, sys) from e


def read_yaml_file(file_path: str) -> dict:
    try:
        logging.info(f"Reading *.yaml file from {file_path}")
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        logging.info(f"Storing *.yaml file to -> {file_path}")
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    try:
        logging.info(f"Storing *.npy file to -> {file_path}")
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open (file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        logging.info(f"Reading *.npy file from {file_path}")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info(f"Storing *.pkl file to -> {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        logging.info(f"Reading *.pkl file from {file_path}")
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    

def fetch_data_from_database(collection_name:str,
                             database_name:str=database)->pd.DataFrame:
    try:
        logging.info(f"Initiate Data Base Connection with: {database_name} collection: {collection_name}")
        mongo_client=MongoClient(mongodb_url, server_api=ServerApi('1'), tlsCAFile=ca)
        collection=mongo_client[database_name][collection_name]
        logging.info("Data base connection established")

        df=pd.DataFrame(list(collection.find()))
        logging.info("Required data retrieved")
        
        if "_id" in list(df.columns):
            df.drop(labels=["_id"], axis=1, inplace=True)

        logging.info("Dataframe import successful.")
        logging.info(f"Loaded {df.shape[0]:,} rows X {df.shape[1]} columns")

        logging.info("Replacing any possible 'na' values in with dataframe with 'np.nan'")
        df.replace({'na': np.nan}, inplace=True)
        
        """Logging a quick statistical summary of the DataFrame."""
        logging.info(f"\n{'='*60}")
        logging.info(f"DATASET SUMMARY")
        logging.info(f"Shape       : {df.shape}")
        logging.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        logging.info(f"Quick Info  :\n{df.info()}")
        logging.info(f"Missing vals:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
        logging.info(f"Dtypes:\n{df.dtypes.to_string()}")
        logging.info(f"{'='*60}")

        return df
    except Exception as e:
        raise CustomException(e, sys) from e
    

def insert_data_into_database(collection_name:str,
                              data:pd.DataFrame=None,
                              database_name:str=database,
                              file_path:str=None) -> None:
    try:
        if file_path:
            df:pd.DataFrame = pd.read_csv(file_path)
        else:
            df=data
        df.reset_index(drop=True, inplace=True)
        records:json = list(json.loads(df.T.to_json()).values())

        logging.info(f"Initiate Data Base Connection with: {database_name} collection: {collection_name}")
        mongo_client=MongoClient(mongodb_url, server_api=ServerApi('1'), tlsCAFile=ca)
        logging.info("Data base connection established")

        collection=mongo_client[database_name][collection_name]
        collection.insert_many(records)
        logging.info(f"{len(records)} records insertion complete")

    except Exception as e:
        raise CustomException(e, sys) from e