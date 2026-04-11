import os, sys, yaml, pickle
from src.exception.exception import CustomException
from src.logging.logger import get_logger
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, List
import matplotlib.pyplot as plt

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